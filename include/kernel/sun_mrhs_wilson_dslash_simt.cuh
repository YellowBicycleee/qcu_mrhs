#pragma once

#include <cstdint>
#include <cub/block/block_load.cuh>
#include <kernel/precondition/eo_precondition.cuh>

#include "base/datatype/qcu_complex.cuh"
#include "base/datatype/qcu_float2.cuh"
#include "desc/qcu_desc.h"
#include "kernel/check_boundary.cuh"
#include "kernel/gemm/qcu_gemm_configure.cuh"
#include "kernel/gemm/qcu_gemm_loader.cuh"
#include "kernel/qcu_gamma.cuh"
#include "kernel/su_n_m_rhs_matmul.cuh"
#include "point/qcu_point.cuh"
#include "qcu_helper.h"
namespace qcu {
namespace device {

/**
* @brief calculate the wilson dslash for a single point
*
* use Batch-gemm like method
*   A is Gauge, is row-col major and col-major in glb memory, col-major in smem, glb_size = n_color * n_color
*   B is fermion in, we calculate 2 * n_color * m_rhs size of projected fermion, row-major in glb and smem
*
*   C is 4 *[m_rhs * n_color] size of projected fermion in reg, 2 * n_color * m_rhs size of temp_res in register
*/
template <
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    bool _use_tensor_core = false,
    int Stages = 1,
    typename Float2 = Float2_t<FloatType_>,
    typename Complex = Complex<FloatType_>
>
QCU_DEVICE
void single_point_wilson_dslash(
    FloatType_* __restrict__ out,
    FloatType_* __restrict__ in,
    FloatType_* __restrict__ gauge,
    QcuLattDesc latt_desc, int multiprocess , int parity,
    bool dagger_flag, int n_color, int m_rhs, int coord_1dim, 
    FloatType_ kappa = 0, bool mat = false)
{
    using GaugeMatShape = gemm::MatShape<BlockShape_::kM, BlockShape_::kK>;
    using FermionMatShape = gemm::MatShape<BlockShape_::kK, BlockShape_::kN>;

    constexpr int A_Shape = GaugeMatShape::kMN;
    constexpr int B_Shape = FermionMatShape::kMN;

    const int fermion_site_length = n_color * m_rhs;
    // used for ping pong
    __shared__ Float2 smem_A[Stages][A_Shape]; // smem A size = BlockShape_::kMK
    __shared__ Float2 smem_B[Stages][B_Shape * 2]; // smem B size = BlockShape_::kKN
    // __shared__ Float2 smem_B2[Stages][BlockShape_::kKN];

    // ldg_A and ldg_B are used to load A and B from global memory
    Complex ldg_A[1]; // BlockShape_::kMK / BlockSize_
    Complex ldg_B[1]; // BlockShape_::kKN / BlockSize_

    Complex temp_res[2][1]; // BlockShape_::kMN / BlockSize_
    Complex res[4][1];

    QcuLattDesc latt_half_desc{latt_desc.X() >> 1, latt_desc.Y(), latt_desc.Z(), latt_desc.T()};

    Point coord {coord_1dim % latt_half_desc.X()
        , coord_1dim % (latt_half_desc.Y() * latt_half_desc.X()) / latt_half_desc.X()
        , coord_1dim % (latt_half_desc.Z() * latt_half_desc.Y() * latt_half_desc.X()) / (latt_half_desc.Y() * latt_half_desc.X())
        , coord_1dim / (latt_half_desc.Z() * latt_half_desc.Y() * latt_half_desc.X())
        , parity
    };

    Point move_coord;

    int32_t mat1_pos; // will be 0 or 1, use this to set mat1 position
    int32_t mat2_pos; // will be 2 or 3, use this to set mat2 position     temp_mat = mat1 + scale * mat2

    int32_t blocks_m = div_ceil(n_color, BlockShape_::kM);
    int32_t blocks_n = div_ceil(m_rhs, BlockShape_::kN);

    Complex scale; // when read B, use B1 + scale B2

    for (int loop_blk_m = blockIdx.y; loop_blk_m < blocks_m; loop_blk_m += gridDim.y) {
        for (int loop_blk_n = blockIdx.x; loop_blk_n < blocks_n; loop_blk_n += gridDim.x) {

            int row = loop_blk_m * BlockShape_::kM;// + threadIdx.y;
            int col = loop_blk_n * BlockShape_::kN;// + threadIdx.x;

            for (int i = 0; i < Ns; ++i) {
                res[i][0] = 0;
            }

            // even some points are out of range, we still need to calculate them,
            // otherwise, deadlock will happen
            for (int dim_dir = 0; dim_dir < Nd * DIRECTIONS; dim_dir++) {
                int dir = dim_dir & 1;  // same with '% DIRECTIONS'
                int dim = dim_dir >> 1; // same with '/ DIRECTIONS'

                move_coord = coord.move(dir, dim, latt_half_desc);

                // calculate start addr of global A and B
                Float2* glb_A;
                Float2* glb_B = reinterpret_cast<Float2 *>(
                    move_coord.getGatheredColorSpinorAddr(in, latt_half_desc, n_color, m_rhs));

                // set dagger, BE CAREFUL: it is possible to be wrong here
                if (dir == FWD) {
                    glb_A = reinterpret_cast<Float2 *>(coord.getGaugeAddr(gauge, dim, latt_half_desc, n_color));
                }
                else { // bwd default: not dagger
                    glb_A = reinterpret_cast<Float2 *>(move_coord.getGaugeAddr(gauge, dim, latt_half_desc, n_color));
                }

                // main loop
                for (int k = 0; k < n_color; k += BlockShape_::kK) {
                    /// load Gauge
                    /// load A from global memory to register, then store to smem
                    if (dir == FWD) { // global memory is row-major, col-major in smem
                        gemm::ldg<Float2, GaugeMatShape, BlockShape_, WarpShape_>
                            ( glb_A, n_color, n_color, row, k, reinterpret_cast<Float2*>(ldg_A));
                        gemm::sts_direct<Float2, gemm::GemmShape<BlockShape_::kM, BlockShape_::kK, 0>, WarpShape_>
                            (smem_A[0], reinterpret_cast<Float2*>(ldg_A));
                    } else {        // global memory is col-major, col-major in smem
                        gemm::ldg<Float2, gemm::MatShapeTranspose<GaugeMatShape>, BlockShape_, WarpShape_>
                            (glb_A, n_color, n_color, k, row, reinterpret_cast<Float2*>(ldg_A));

                        // dagger
                        for (int i = 0; i < sizeof(ldg_A) / sizeof(Complex); i++) {
                            ldg_A[i] = ldg_A[i].conj();
                        }

                        gemm::sts_transpose<Float2, GaugeMatShape, BlockShape_, WarpShape_> (smem_A[0], reinterpret_cast<Float2*>(ldg_A));
                    }
                    __syncthreads();

                    /// load Fermion
                    /// load B from global memory to register, need combine 2 of 4 in global memory to 2 in smem
                    #pragma unroll
                    for (int pos = 0; pos < 2; ++pos) {
                        if (row < n_color && col < m_rhs) {
                            mat1_pos = pos;
                            mat2_pos = kernel::Gamma<FloatType_>::get_reconstruct_mat_id(dim, mat1_pos);
                            // get scale
                            scale = kernel::Gamma<FloatType_>::get_projection_scale(dim, mat1_pos, dir);
                            if (dagger_flag) { scale = -scale; }
                        }
                        gemm::ldg_fermion<FloatType_, FermionMatShape, BlockShape_, WarpShape_> (
                            reinterpret_cast<FloatType_*>(glb_B + mat1_pos * fermion_site_length),
                            reinterpret_cast<FloatType_*>(glb_B + mat2_pos * fermion_site_length),
                            n_color, m_rhs, scale, k, col, reinterpret_cast<Float2_t<FloatType_> *>(ldg_B));
                        gemm::sts_direct<Float2, FermionMatShape, BlockShape_, WarpShape_>
                                (&smem_B[0][pos * B_Shape], reinterpret_cast<Float2*>(ldg_B));
                    }

                    __syncthreads();

                    // gemm, MMA
                    temp_res[0][0] = 0;
                    temp_res[1][0] = 0;
                    for (int kk = 0; kk < BlockShape_::kK; ++kk) {
                        Float2 a  = smem_A[0][threadIdx.y * BlockShape_::kK + kk];
                        Float2 b1 = smem_B[0][0 * B_Shape + kk * BlockShape_::kN + threadIdx.x];
                        Float2 b2 = smem_B[0][1 * B_Shape + kk * BlockShape_::kN + threadIdx.x];
                        temp_res[0][0] += Complex(a) * Complex(b1);
                        temp_res[1][0] += Complex(a) * Complex(b2);
                    }
                    __syncthreads();
                    // add to res
                    if (row < n_color && col < m_rhs) {
                        for (mat1_pos = 0; mat1_pos < 2; ++mat1_pos) {
                            mat2_pos = kernel::Gamma<FloatType_>::get_reconstruct_mat_id(dim, mat1_pos);
                            scale = kernel::Gamma<FloatType_>::get_reconstruct_scale(dim, mat1_pos, dir);

                            if (dagger_flag) { scale = -scale; }

                            res[mat1_pos][0] += temp_res[mat1_pos][0];
                            res[mat2_pos][0] += scale * temp_res[mat1_pos][0]; // scale calculated from 1 + gamma, so need to add '-'
                        }
                    }
                } // end main loop for
            }

            // store res to global memory
            Float2* glb_out = reinterpret_cast<Float2 *>(coord.getGatheredColorSpinorAddr(out, latt_half_desc, n_color, m_rhs));

            // store global memory
#pragma unroll
            for (int i = 0; i < Nd; ++i) {
                gemm::stg<Float2, FermionMatShape, BlockShape_, WarpShape_> (
                    reinterpret_cast<Float2*>(glb_out) + i * n_color * m_rhs,
                    n_color, m_rhs, row, col,
                    reinterpret_cast<Float2*>(res[i]));
            }
        }
    }
}

// entry function
// parity is the parity of the point of fermion out,
// 1 - parity is the parity of the point of fermion in
template <
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    bool _use_tensor_core = false
>
QCU_GLOBAL
void wilson_dslash_su_n_mrhs(
    FloatType_* __restrict__ out,
    FloatType_* __restrict__ in,
    FloatType_* __restrict__ gauge,
    QcuLattDesc latt_desc, int multiprocess,
    int parity, bool dagger_flag, int n_color, int m_rhs) 
{
    assert(BlockShape_::kM > 0 && BlockShape_::kN > 0 && BlockShape_::kK > 0);

    // z 轴切分矩阵坐标点，(x,y)切分单个矩阵
    int block_id = blockIdx.z;
    int grid_size = gridDim.z;  // 1D grid 
    int half_vol = latt_desc.half_lattice_volume();

    for (int i = block_id; i < half_vol; i += grid_size) {
        single_point_wilson_dslash<FloatType_, BlockShape_, WarpShape_>(
            out, in, gauge, latt_desc, multiprocess, parity, dagger_flag, n_color, m_rhs, i);
    }
}

}  // namespace device
}  // namespace qcu