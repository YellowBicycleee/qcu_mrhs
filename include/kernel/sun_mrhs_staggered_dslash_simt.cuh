//
// Created by wangj on 2025/2/13.
//

#pragma once
#include "qcu_helper.h"
#include "kernel/gemm/qcu_gemm_configure.cuh"
#include "kernel/gemm/qcu_gemm_loader.cuh"
#include "point/qcu_point.cuh"

namespace qcu::device {

template <
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    int Nspin_ = 1,
    bool use_tensor_core_ = false,
    int Stages_ = 1,
    typename Float2 = Float2_t<FloatType_>,
    typename Complex = qcu::Complex<FloatType_>
>
class StaggeredDslashDevice {

public:
    struct Argument {
        bool dagger_flag;
        FloatType_* __restrict__ out_half;
        FloatType_* __restrict__ in_half;
        FloatType_* __restrict__ gauge;
        QcuLattDesc* latt_desc;
        unsigned int multiprocess;
        int parity;
        int n_color;
        int m_rhs;
        int coord_1dim;
        FloatType_ kappa = 0;
    };

    QCU_DEVICE void operator ()(Argument& arg, bool mat_flag = false) {

        using GaugeMatShape = gemm::MatShape<BlockShape_::kM, BlockShape_::kK>;
        using FermionMatShape = gemm::MatShape<BlockShape_::kK, BlockShape_::kN>;

        constexpr int A_Shape = GaugeMatShape::kMN;
        constexpr int B_Shape = FermionMatShape::kMN;

        const int fermion_site_length = arg.n_color * arg.m_rhs;
        // used for ping pong
        __shared__ Float2 smem_A[Stages_][A_Shape]; // smem A size = BlockShape_::kMK
        __shared__ Float2 smem_B[Stages_][B_Shape * Nspin_]; // smem B size = BlockShape_::kKN

        // ldg_A and ldg_B are used to load A and B from global memory
        Complex ldg_A[Stages_]; // BlockShape_::kMK / BlockSize_
        Complex ldg_B[Stages_]; // BlockShape_::kKN / BlockSize_

        Complex temp_res[Nspin_]; // BlockShape_::kMN / BlockSize_
        Complex res[Nspin_];

        QcuLattDesc latt_half_desc{arg.latt_desc.X() >> 1, arg.latt_desc.Y(), arg.latt_desc.Z(), arg.latt_desc.T()};

        Point<Nspin_> coord {arg.coord_1dim % latt_half_desc.X()
            , arg.coord_1dim % (latt_half_desc.Y() * latt_half_desc.X()) / latt_half_desc.X()
            , arg.coord_1dim % (latt_half_desc.Z() * latt_half_desc.Y() * latt_half_desc.X()) / (latt_half_desc.Y() * latt_half_desc.X())
            , arg.coord_1dim / (latt_half_desc.Z() * latt_half_desc.Y() * latt_half_desc.X())
            , arg.parity
        };

        Point<Nspin_> move_coord;

        int32_t blocks_m = div_ceil(arg.n_color, BlockShape_::kM);
        int32_t blocks_n = div_ceil(arg.m_rhs, BlockShape_::kN);

        for (int loop_blk_m = blockIdx.y; loop_blk_m < blocks_m; loop_blk_m += gridDim.y) {
            for (int loop_blk_n = blockIdx.x; loop_blk_n < blocks_n; loop_blk_n += gridDim.x) {

                int row = loop_blk_m * BlockShape_::kM;// + threadIdx.y;
                int col = loop_blk_n * BlockShape_::kN;// + threadIdx.x;

                for (int i = 0; i < Nspin_; ++i) {  res[i][0] = 0; }

                // even some points are out of range, we still need to calculate them, otherwise, deadlock will happen
                for (int dim_dir = 0; dim_dir < Nd * DIRECTIONS; dim_dir++) {

                    int dir = dim_dir & 1;  // same with '% DIRECTIONS'
                    int dim = dim_dir >> 1; // same with '/ DIRECTIONS'

                    // for boundary check
                    if (arg.multiprocess & (1 << dim)) {
                        int cb = (coord.Y() + coord.Z() + coord.T()) % 2;
                        if (dim == X_DIM) {
                            if ((dir == FWD && coord.X() == latt_half_desc.X() - 1 && cb != arg.parity) ||
                                (dir == BWD && coord.X() == 0 && cb == arg.parity))
                            { continue; }
                        }
                        else {
                            if ((dir == FWD && coord.at(dim) == latt_half_desc.at(dim) - 1)
                                || (dir == BWD && coord.at(dim) == 0))
                            {   continue; }
                        }
                    }

                    move_coord = coord.move(dir, dim, latt_half_desc);

                    // calculate start addr of global A and B
                    Float2* glb_A;
                    Float2* glb_B = reinterpret_cast<Float2 *>(
                        move_coord.getGatheredColorSpinorAddr(arg.in, latt_half_desc, arg.n_color, arg.m_rhs));

                    // set dagger, BE CAREFUL: it is possible to be wrong here
                    if (dir == FWD) {
                        glb_A = reinterpret_cast<Float2 *>(coord.getGaugeAddr(arg.gauge, dim, latt_half_desc, arg.n_color));
                    }
                    else { // bwd default: not dagger
                        glb_A = reinterpret_cast<Float2 *>(move_coord.getGaugeAddr(arg.gauge, dim, latt_half_desc, arg.n_color));
                    }

                    // main loop
                    for (int k = 0; k < arg.n_color; k += BlockShape_::kK) {
                        /// load Gauge
                        /// load A from global memory to register, then store to smem
                        if (dir == FWD) { // global memory is row-major, col-major in smem

                            gemm::ldg<Float2, GaugeMatShape, BlockShape_, WarpShape_>
                                ( glb_A, arg.n_color, arg.n_color, row, k, reinterpret_cast<Float2*>(ldg_A));
                            gemm::sts_direct<Float2, GaugeMatShape, BlockShape_, WarpShape_>
                                (smem_A[0], reinterpret_cast<Float2*>(ldg_A));

                        } else {        // global memory is col-major, col-major in smem

                            gemm::ldg<Float2, gemm::MatShapeTranspose<GaugeMatShape>, BlockShape_, WarpShape_>
                                (glb_A, arg.n_color, arg.n_color, k, row, reinterpret_cast<Float2*>(ldg_A));
                            // dagger
                            for (int i = 0; i < sizeof(ldg_A) / sizeof(Complex); i++) { ldg_A[i] = ldg_A[i].conj(); }

                            gemm::sts_transpose<Float2, GaugeMatShape, BlockShape_, WarpShape_> (smem_A[0], reinterpret_cast<Float2*>(ldg_A));

                        }
                        __syncthreads();

                        // load Fermion, load B from global memory to register
                        #pragma unroll
                        for (int pos = 0; pos < Nspin_; ++pos) {
                            gemm::ldg<Float2, gemm::MatShapeTranspose<FermionMatShape>, BlockShape_, WarpShape_>(
                                glb_B, arg.n_color, arg.m_rhs, k, col, reinterpret_cast<Float2*>(ldg_B)
                            );
                        }

                        __syncthreads();

                        // gemm, MMA
                        for (int kk = 0; kk < BlockShape_::kK; ++kk) {
                            Float2 a  = smem_A[0][threadIdx.y * BlockShape_::kK + kk];
                            Float2 b_arr[Nspin_];
                            for (int pos = 0; pos < Nspin_; ++pos) {
                                b_arr[pos] = smem_B[0][pos * B_Shape + kk * BlockShape_::kN + threadIdx.x];
                                temp_res[pos] = Complex(a) * Complex(b_arr[pos]);
                            }
                        }
                        __syncthreads();
                        // add to res
                        if (row < arg.n_color && col < arg.m_rhs) {
                            #pragma unroll
                            for (int i = 0; i < Nspin_; ++i) {
                                res[i] += temp_res[i];
                            }
                        }
                    } // end main loop for
                }

                // store res to global memory
                Float2* glb_out = reinterpret_cast<Float2 *>(
                    coord.getGatheredColorSpinorAddr(arg.out, latt_half_desc, arg.n_color, arg.m_rhs));

                // store global memory
    #pragma unroll
                for (int i = 0; i < Nspin_; ++i) {
                    gemm::stg<Float2, FermionMatShape, BlockShape_, WarpShape_> (
                        reinterpret_cast<Float2*>(glb_out) + i * arg.n_color * arg.m_rhs,
                        arg.n_color, arg.m_rhs, row, col, reinterpret_cast<Float2*>(res[i]));
                }
            }
        }
    }
};


// entry function
// parity is the parity of the point of fermion out,
// 1 - parity is the parity of the point of fermion in
template <
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    int Nspin_ = 1,
    bool TensorOpEnabled_ = false
>
QCU_GLOBAL
void staggered_dslash_su_n_mrhs(
    FloatType_* __restrict__ out_half,
    FloatType_* __restrict__ in_half,
    FloatType_* __restrict__ gauge,
    QcuLattDesc latt_desc, unsigned int multiprocess,
    int parity, bool dagger_flag, int n_color, int m_rhs,
    FloatType_ kappa = 0,
    bool mat_flag = false)
{
    using Argument = typename StaggeredDslashDevice<FloatType_, BlockShape_, WarpShape_, Nspin_, TensorOpEnabled_>::Argument;

    assert(BlockShape_::kM > 0 && BlockShape_::kN > 0 && BlockShape_::kK > 0);

    // z 轴切分矩阵坐标点，(x,y)切分单个矩阵
    int block_id = blockIdx.z;
    int grid_size = gridDim.z;  // 1D grid
    int half_vol = latt_desc.half_lattice_volume();

    StaggeredDslashDevice<FloatType_, BlockShape_, WarpShape_, Nspin_, TensorOpEnabled_> stagggered_op;
    for (int i = block_id; i < half_vol; i += grid_size) {
        Argument arg {
            .dagger_flag = dagger_flag,
            .out_half = out_half,
            .in_half = in_half,   // 计算out-even的时候，in为odd
            .gauge = gauge,
            .latt_desc = &latt_desc,
            .multiprocess = multiprocess,
            .parity = parity,
            .n_color = n_color,
            .m_rhs = m_rhs,
            .coord_1dim = i,
            .kappa = kappa
        };

        stagggered_op(arg, mat_flag);
    }
}


}