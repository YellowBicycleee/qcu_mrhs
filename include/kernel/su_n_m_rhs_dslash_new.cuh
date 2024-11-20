#pragma once

#include <cstdint>
#include <cub/block/block_load.cuh>

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
    int BlockSize_ = 128,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    bool _use_tensor_core = false
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
    const int fermion_site_length = n_color * m_rhs * 2    ;
    // used for ping pong
    __shared__ Float2_t<FloatType_> smem_A[2][BlockShape_::kMK]; // smem A size = BlockShape_::kMK
    __shared__ Float2_t<FloatType_> smem_B[2][BlockShape_::kKN]; // smem B size = BlockShape_::kKN
    // C size = _BLK_M * _BLK_N in register and Res size is _BLK_N * _BLK_N * 2 in register

    // ldg_A and ldg_B are used to load A and B from global memory
    Complex<FloatType_> ldg_A[2][BlockShape_::kMK / BlockSize_];
    Complex<FloatType_> ldg_B[2][BlockShape_::kKN / BlockSize_];

    Complex<FloatType_> temp_res[BlockShape_::kMN / BlockSize_];
    Complex<FloatType_> res1[BlockShape_::kMN * 2 / BlockSize_];
    Complex<FloatType_> res2[BlockShape_::kMN * 2 / BlockSize_];
    int half_Lx = (latt_desc.X() << 1);

    Point coord { coord_1dim / (latt_desc.Z() * latt_desc.Y() * half_Lx)
                , coord_1dim % (latt_desc.Z() * latt_desc.Y() * half_Lx) / (latt_desc.Y() * half_Lx)
                , coord_1dim % (latt_desc.Y() * half_Lx) / half_Lx
                , coord_1dim % half_Lx
                , parity};
    Point move_coord;

    int32_t mat1_pos; // will be 0 or 1, use this to set mat1 position
    int32_t mat2_pos; // will be 2 or 3, use this to set mat2 position     temp_mat = mat1 + scale * mat2

    int32_t blocks_m = div_ceil(n_color, BlockShape_::kM);
    int32_t blocks_n = div_ceil(2 * m_rhs, BlockShape_::kN);

    Float2_t<FloatType_> scale; // when read B, use B1 + scale B2

    for (int loop_blk_m = 0; loop_blk_m < blocks_m; ++loop_blk_m) {
        for (int loop_blk_n = 0; loop_blk_n < blocks_n; ++loop_blk_n) {

            int row = loop_blk_m * BlockShape_::kM + threadIdx.y;
            int col = loop_blk_n * BlockShape_::kN + threadIdx.x;

            if (row < n_color && col < 2 * m_rhs) {
                mat1_pos = col / m_rhs;

                for (int dim_dir = 0; dim_dir < Nd * DIRECTIONS; dim_dir++) {
                    int dir = dim_dir & 1;  // same with '% DIRECTIONS'
                    int dim = dim_dir >> 1; // same with '/ DIRECTIONS'

                    mat2_pos = kernel::Gamma<FloatType_>::get_reconstruct_mat_id(dim, mat1_pos);
                    // get scale
                    scale = kernel::get_scale<FloatType_>(dim, mat1_pos);

                    move_coord = coord.move(dir, dim, half_Lx, latt_desc.Y(), latt_desc.Z(), latt_desc.T());
                    // calculate start addr of global A and B
                    Float2_t<FloatType_>* glb_A;
                    Float2_t<FloatType_>* glb_B = move_coord.getGatheredColorSpinorAddr(in, half_Lx, latt_desc.Y(),
                        latt_desc.Z(), latt_desc.T(), n_color, m_rhs);

                    // set dagger, BE CAREFUL: it is possible to be wrong here
                    if (dir == FWD) { // fwd default: dagger
                        glb_A = coord.getGaugeAddr(gauge, dim, half_Lx, latt_desc.Y(), latt_desc.Z(), latt_desc.T());
                        if (!dagger_flag) {
                            scale = -scale;
                        }
                    } else { // bwd default: not dagger
                        glb_A = move_coord.getGaugeAddr(gauge, dim, half_Lx, latt_desc.Y(), latt_desc.Z(), latt_desc.T());
                        if (dagger_flag) {
                            scale = -scale;
                        }
                    }

                    // main loop
                    for (int k = 0; k < n_color; k += BlockShape_::kK) {
                        // load A from global memory to register, then store to smem
                        if (dir == FWD) { // global memory is row-major, col-major in smem
                            gemm::ldg<Float2_t<FloatType_>, BlockShape_, WarpShape_>>(glb_A, n_color, n_color, row, col, ldg_A[0]);
                            gemm::sts_direct<Float2_t<FloatType_>,
                                    gemm::GemmShape<BlockShape_::kM, BlockShape_::kK, 0>,
                                    WarpShape_>
                                >(smem_A, ldg_A[0]);
                        } else {        // global memory is col-major, col-major in smem
                            gemm::ldg<Float2_t<FloatType_>, gemm::GemmShapeTranspose<BlockShape_>, WarpShape_>>
                                (glb_A, n_color, n_color, col, row, ldg_A[0]);
                            gemm::sts_transpose<Float2_t<FloatType_>,
                                    gemm::GemmShape<BlockShape_::kM, BlockShape_::kK, 0>,
                                    WarpShape_>
                                >(smem_A, ldg_A[0]);
                        }

                        // load B from global memory to register, need combine 2 of 4 in global memory to 2 in smem
                        gemm::ldg_fermion<FloatType_, gemm::GemmShape<BlockShape_::kK, BlockShape_::kN, 0>, WarpShape_> (
                            glb_B + mat1_pos * fermion_site_length * 2, glb_B + mat2_pos * fermion_site_length * 2, /*float->complex<Float>*/
                            n_color, 2 * m_rhs, scale, k, col, ldg_B[0]);
                        // gemm, MMA
                        temp_res[0][0] = 0;
                        for (int kk = 0; kk < BlockShape_::kK; ++kk) {
                            Complex<FloatType_> a = smem_A[threadIdx.y * BlockShape_::kK + kk];
                            Complex<FloatType_> b = smem_B[kk * BlockShape_::kN + threadIdx.x];
                            temp_res += a * b;
                        }

                        // add to res
                        scale = kernel::Gamma<FloatType_>::get_reconstruct_scale(dim, mat1_pos);
                        res1[0][0] += temp_res[0][0];
                        res2[0][0] += temp_res[0][0] * scale;
                    }
                }
                // store res to global memory
                Float2_t<FloatType_>* glb_out = coord.getGatheredColorSpinorAddr(out, half_Lx, latt_desc.Y(),
                        latt_desc.Z(), latt_desc.T(), n_color, m_rhs);

                // we calculated [row ~ row + BlockShape_::kM, col ~ col + BlockShape_::kN]
                gemm::stg<Float2_t<FloatType_>, BlockShape_, WarpShape_> (
                    reinterpret_cast<Float2_t<FloatType_>*>(glb_out),
                    n_color, 2 * m_rhs, row, col, reinterpret_cast<Float2_t<FloatType_>*>(res1[0]));
                gemm::stg<Float2_t<FloatType_>, BlockShape_, WarpShape_> (
                    reinterpret_cast<Float2_t<FloatType_>*>(glb_out) + 2 * n_color * m_rhs,
                    n_color, 2 * m_rhs, row, col, reinterpret_cast<Float2_t<FloatType_>*>(res1[0]));
            }

        }
    }
}

// entry function
// parity is the parity of the point of fermion out,
// 1 - parity is the parity of the point of fermion in
template <
    typename FloatType_ = double,
    int BlockSize_ = 128,
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
    // z 轴切分矩阵坐标点，(x,y)切分单个矩阵
    int block_id = blockIdx.z;
    int grid_size = gridDim.z;  // 1D grid
    int half_vol = latt_desc.half_lattice_volume();

    for (int i = block_id; i < half_vol; i += grid_size) {
        single_point_wilson_dslash<FloatType_, BlockSize_, BlockShape_, WarpShape_>(
            out, in, gauge, latt_desc, multiprocess, parity, dagger_flag, n_color, m_rhs, i);
    }

}

}  // namespace device
}  // namespace qcu