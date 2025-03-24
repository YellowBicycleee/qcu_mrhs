#pragma once
#include "qcu_helper.h"
#include "kernel/gemm/qcu_gemm_configure.cuh"
#include "kernel/qcu_gamma.cuh"
#include "kernel/qcu_float4.cuh"
#include "point/qcu_point.cuh"
#include "qcu_utils.h"
#include "complex/qcu_complex.cuh"
#include "sun_mrhs_wilson_dslash_load_gauge_fermion.cuh"
#include <mma.h>
namespace qcu::device::tensorop {

using namespace nvcuda;

template <
    typename Float_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    int Nspin_ = 4,
    bool use_tensor_core_ = true,
    int Stages_ = 1,
    typename Float2_ = Float2_t<Float_>,
    typename Complex_ = qcu::Complex<Float_>
>
class WilsonDslashDevice {

public:
    struct Argument {
        bool dagger_flag;
        Float_* __restrict__ out_half;
        Float_* __restrict__ in_half;
        Float_* __restrict__ gauge;
        QcuLattDesc latt_desc;
        unsigned int multiprocess;
        int parity;
        int n_color;
        int m_rhs;
        int coord_1dim;
        Float_ kappa = 0;
        int t_boundary = 1;
    };

    QCU_DEVICE void operator ()(Argument& arg, bool mat_flag = false) {


        assert(BlockShape_::kMN / kWarpSize == blockDim.x * blockDim.y);
        assert(BlockShape_::kM >= WarpShape_::kM && BlockShape_::kN >= WarpShape_::kN && BlockShape_::kK >= WarpShape_::kK);
        const int fermion_site_length = arg.n_color * arg.m_rhs;
        constexpr int kElemsPerThread = WarpShape_::kMN / kWarpSize;
        __shared__ Float_ A_tile_real[BlockShape_::kMK];
        __shared__ Float_ A_tile_imag[BlockShape_::kMK];
        __shared__ Float_ B_tile_real[2][BlockShape_::kKN]; // Ns / 2
        __shared__ Float_ B_tile_imag[2][BlockShape_::kKN];

        __shared__ Float_ C_tile_store_buf_real[CTileShape::kMN];
        __shared__ Float_ C_tile_store_buf_imag[CTileShape::kMN];
        // Complex_ ldg_A, ldg_B;
        Complex_ result[4][kElemsPerThread];// = {0}; // Nspin
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < kElemsPerThread; j++) {
                result[i][j] = Complex_(0, 0);
            }
        }

        QcuLattDesc latt_half_desc{arg.latt_desc.X() >> 1, arg.latt_desc.Y(), arg.latt_desc.Z(), arg.latt_desc.T()};

        Point<Nspin_> coord {arg.coord_1dim % latt_half_desc.X()
            , arg.coord_1dim % (latt_half_desc.Y() * latt_half_desc.X()) / latt_half_desc.X()
            , arg.coord_1dim % (latt_half_desc.Z() * latt_half_desc.Y() * latt_half_desc.X()) / (latt_half_desc.Y() * latt_half_desc.X())
            , arg.coord_1dim / (latt_half_desc.Z() * latt_half_desc.Y() * latt_half_desc.X())
            , arg.parity
        };

        Point<Nspin_> move_coord;

        int mat1_pos, mat2_pos; // mat1_pos will be 0 or 1, mat2_pos will be 2 or 3,   temp_mat = mat1 + scale * mat2

        int blocks_m = div_ceil(arg.n_color, BlockShape_::kM);
        int blocks_n = div_ceil(arg.m_rhs, BlockShape_::kN);
        int warp_rank = (threadIdx.y * blockDim.x + threadIdx.x) / kWarpSize;
        int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % kWarpSize;
        int warp_rank_row = warp_rank / kWarpNumCol;
        int warp_rank_col = warp_rank % kWarpNumCol;
        Complex_ scale; // when read B, use B1 + scale B2

        for (int loop_blk_m = blockIdx.y; loop_blk_m < blocks_m; loop_blk_m += gridDim.y) {
            for (int loop_blk_n = blockIdx.x; loop_blk_n < blocks_n; loop_blk_n += gridDim.x) {

                int block_row = loop_blk_m * BlockShape_::kM, block_col = loop_blk_n * BlockShape_::kN;

                wmma::fragment<wmma::accumulator, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_> temp_r[2];
                wmma::fragment<wmma::accumulator, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_> temp_i[2];

                for (int dim = X_DIM; dim < Nd; ++dim) {
#pragma unroll
                    for (int dir = 0; dir < DIRECTIONS; ++dir) {
                        // for boundary check
                        if (arg.multiprocess & (1 << dim)) {
                            int cb = (coord.Y() + coord.Z() + coord.T()) % 2;
                            if (dim == X_DIM) {
                                if ((dir == FWD && coord.X() == latt_half_desc.X() - 1 && cb != arg.parity) || (dir == BWD && coord.X() == 0 && cb == arg.parity)) { continue; }
                            }
                            else {
                                if ((dir == FWD && coord.at(dim) == latt_half_desc.at(dim) - 1) || (dir == BWD && coord.at(dim) == 0)) {  continue; }
                            }
                        }

                        move_coord = coord.move(dir, dim, latt_half_desc);

                        Float2_* glb_A; // calculate start addr of global A and B
                        // set dagger, BE CAREFUL: it is possible to be wrong here
                        if (dir == FWD) {
                            glb_A = reinterpret_cast<Float2_ *>(coord.getGaugeAddr(arg.gauge, dim, latt_half_desc, arg.n_color));
                        }
                        else { // bwd default: not dagger
                            glb_A = reinterpret_cast<Float2_ *>(move_coord.getGaugeAddr(arg.gauge, dim, latt_half_desc, arg.n_color));
                        }

                        Float2_* glb_B = reinterpret_cast<Float2_ *>(move_coord.getGatheredColorSpinorAddr(arg.in_half, latt_half_desc, arg.n_color, arg.m_rhs));


                        // wmma::fragment<wmma::accumulator, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_> temp_r[2];
                        // wmma::fragment<wmma::accumulator, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_> temp_i[2];
                        for (int pos = 0; pos < 2; ++pos) {
                            wmma::fill_fragment(temp_r[pos], 0.0f);
                            wmma::fill_fragment(temp_i[pos], 0.0f);
                        }
                        // k-loop
                        for (int k = 0; k < arg.n_color; k += BlockShape_::kK) {
                            // ldg Gauge
                            if (dir == FWD) { // global memory is row-major, col-major in smem
                                ldg_and_sts<Float_, GaugeMatShape>(glb_A, block_row, k, arg.n_color, arg.n_color, A_tile_real, A_tile_imag);
                            } else {        // global memory is col-major, col-major in smem
                                ldg_and_dagger_to_sts<Float_, GaugeMatShape>(glb_A, k, block_row, arg.n_color, arg.n_color, A_tile_real, A_tile_imag);
                            }
                            // __syncthreads();
                            // ldg Fermion
                            #pragma unroll
                            for (int pos = 0; pos < 2; ++pos) {
                                if (block_row < arg.n_color && block_col < arg.m_rhs) {
                                    mat1_pos = pos;
                                    mat2_pos = kernel::Gamma<Float_>::get_reconstruct_mat_id(dim, mat1_pos);
                                    // get scale
                                    scale = kernel::Gamma<Float_>::get_projection_scale(dim, mat1_pos, dir);
                                    if (arg.dagger_flag) { scale = -scale; }
                                }

                                ldg_fermion_sts<Float_, FermionMatShape>(
                                    glb_B + mat1_pos * fermion_site_length,
                                    glb_B + mat2_pos * fermion_site_length,
                                    k, block_col, arg.n_color, arg.m_rhs,
                                    scale, &(B_tile_real[pos][0]), &(B_tile_imag[pos][0])
                                    );
                                    //flag, 2, 0);
                            }
                            __syncthreads();

                            // mma
                            wmma::fragment<wmma::matrix_a, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_, wmma::row_major> a_r_frag;
                            wmma::fragment<wmma::matrix_a, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_, wmma::row_major> a_i_frag;


                            int warp_row_in_blk_left = warp_rank_row * WarpShape_::kM;
                            int warp_col_in_blk_left = k;
                            wmma::load_matrix_sync(a_r_frag, A_tile_real + (warp_row_in_blk_left * BlockShape_::kK + warp_col_in_blk_left), BlockShape_::kK);
                            wmma::load_matrix_sync(a_i_frag, A_tile_imag + (warp_row_in_blk_left * BlockShape_::kK + warp_col_in_blk_left), BlockShape_::kK);
                            for (int pos = 0; pos < 2; ++pos) {
                                wmma::fragment<wmma::matrix_b, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_, wmma::row_major> b_r_frag;
                                wmma::fragment<wmma::matrix_b, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_, wmma::row_major> b_i_frag;
                                warp_row_in_blk_left = k;
                                warp_col_in_blk_left = warp_rank_col * WarpShape_::kN;
                                wmma::load_matrix_sync(b_r_frag, &(B_tile_real[pos][warp_row_in_blk_left * BlockShape_::kN + warp_col_in_blk_left]), BlockShape_::kN);
                                wmma::load_matrix_sync(b_i_frag, &(B_tile_imag[pos][warp_row_in_blk_left * BlockShape_::kN + warp_col_in_blk_left]), BlockShape_::kN);
                                // (ad + bc)i
                                wmma::mma_sync(temp_i[pos], a_r_frag, b_i_frag, temp_i[pos]);
                                wmma::mma_sync(temp_i[pos], a_i_frag, b_r_frag, temp_i[pos]);
                                // (ac - bd) , 注意这里千万不要改A的内容，因为后面还要用到
                                wmma::mma_sync(temp_r[pos], a_r_frag, b_r_frag, temp_r[pos]);
                                for (int elem_pos = 0; elem_pos < b_i_frag.num_elements; elem_pos++) {
                                    b_i_frag.x[elem_pos] = -b_i_frag.x[elem_pos];
                                }
                                wmma::mma_sync(temp_r[pos], a_i_frag, b_i_frag, temp_r[pos]);
                            }
                            __syncthreads();
                        } // end k-loop
                        // add to result;
                        if (block_row < arg.n_color && block_col < arg.m_rhs) {
                            for (mat1_pos = 0; mat1_pos < 2; ++mat1_pos) {
                                mat2_pos = kernel::Gamma<Float_>::get_reconstruct_mat_id(dim, mat1_pos);
                                scale = kernel::Gamma<Float_>::get_reconstruct_scale(dim, mat1_pos, dir);

                                if (arg.dagger_flag) { scale = -scale; }

                                for (int elem_idx = 0; elem_idx < kElemsPerThread; ++elem_idx) {
                                    Complex_ temp_res(temp_r[mat1_pos].x[elem_idx], temp_i[mat1_pos].x[elem_idx]);
                                    result[mat1_pos][elem_idx] += temp_res;
                                    result[mat2_pos][elem_idx] += scale * temp_res; // scale calculated from 1 + gamma, so need to add '-'
                                }
                            }
                        }
                    } // end dir-loop

                } // end dim-loop
                int warp_row_offset = warp_rank_row * WarpShape_::kM;
                int warp_col_offset = warp_rank_col * WarpShape_::kN;
                Float2_* glb_out = reinterpret_cast<Float2_ *>(coord.getGatheredColorSpinorAddr(arg.out_half, latt_half_desc, arg.n_color, arg.m_rhs));
                #pragma unroll
                // store thread result to global memory
                for (int i = 0; i < Nspin_; ++i) { // store global memory
                    Float2_* start = reinterpret_cast<Float2_*>(glb_out) + i * arg.n_color * arg.m_rhs;
                    for (int idx = 0; idx < kElemsPerThread; ++idx) {
                        // 借助temp_r和temp_i来存储结果到smem，再从smem合并存储到glb
                        temp_r[0].x[idx] = result[i][idx].real();
                        temp_i[0].x[idx] = result[i][idx].imag();

                        // if (arg.coord_1dim == 0 && arg.parity == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                        //     // std::cout << "(" <<  << ", " <<  << ")" <;
                        //     printf("(%e,%e)", (double)(temp_r[0].x[idx]), (double)(temp_i[0].x[idx]));
                        // }

                    }
                    wmma::store_matrix_sync(&C_tile_store_buf_real[warp_row_offset * CTileShape::kN + warp_col_offset], temp_r[0], BlockShape_::kN, wmma::mem_row_major);
                    wmma::store_matrix_sync(&C_tile_store_buf_imag[warp_row_offset * CTileShape::kN + warp_col_offset], temp_i[0], BlockShape_::kN, wmma::mem_row_major);
                    __syncthreads();
                    store_matrix_from_smem<Float_, CTileShape>(start, block_row, block_col, arg.n_color, arg.m_rhs, C_tile_store_buf_real, C_tile_store_buf_imag);
                    __syncthreads();

                    // Float2_* start = reinterpret_cast<Float2_*>(glb_out) + i * arg.n_color * arg.m_rhs;
                    // // using F4 = Float4::Float4_t<Float_>;
                    //
                    // int groupId = (lane_id >> 2);
                    // int threadID_in_group = lane_id % 4;
                    // //
                    // int row, col;
                    // for (int idx = 0; idx < kElemsPerThread; ++idx) {
                    // // for (int idx = 0; idx < kElemsPerThread; idx += 2) {
                    //     if (idx < 2 || (idx >= 4 && idx < 6)) {
                    //         row = groupId;
                    //     } else {
                    //         row = groupId + 8;
                    //     }
                    //     if (idx < 4) {
                    //         col = (threadID_in_group * 2) + (idx & 0x1);
                    //     }
                    //     else {
                    //         col = (threadID_in_group * 2) + (idx & 0x1) + 8;
                    //     }
                    //     int row_in_global = block_row + warp_row_offset + row;
                    //     int col_in_global = block_col + warp_col_offset + col;
                    //     if (row_in_global < arg.n_color && col_in_global < arg.m_rhs) {
                    //         start[row_in_global * arg.m_rhs + col_in_global]
                    //                 = reinterpret_cast<Float2_*>(&(result[i][0]))[idx];
                    //     }
                    // }
                }
            }
        }
    }

    QCU_DEVICE WilsonDslashDevice (int n_color, int m_rhs)
        : blocks_m (div_ceil(n_color, BlockShape_::kM))
        , blocks_n (div_ceil(m_rhs, BlockShape_::kN))
    {}

private:
    using GaugeMatShape = gemm::MatShape<BlockShape_::kM, BlockShape_::kK>;
    using FermionMatShape = gemm::MatShape<BlockShape_::kK, BlockShape_::kN>;
    using CTileShape = gemm::MatShape<BlockShape_::kM, BlockShape_::kN>;

    static constexpr int A_Shape = GaugeMatShape::kMN;
    static constexpr int B_Shape = FermionMatShape::kMN;

    static constexpr int kWarpNumRow = BlockShape_::kM / WarpShape_::kM;
    static constexpr int kWarpNumCol = BlockShape_::kN / WarpShape_::kN;

    const int blocks_m;
    const int blocks_n;
};


// entry function
// parity is the parity of the point of fermion out,
// 1 - parity is the parity of the point of fermion in
template <
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    int Nspin_ = 4,
    bool TensorOpEnabled_ = true
>
QCU_GLOBAL
void wilson_dslash_su_n_mrhs(
    FloatType_* __restrict__ out_half,
    FloatType_* __restrict__ in_half,
    FloatType_* __restrict__ gauge,
    QcuLattDesc latt_desc,
    unsigned int multiprocess,
    int parity,
    bool dagger_flag,
    int n_color,
    int m_rhs,
    FloatType_ kappa = 0,
    bool mat_flag = false,
    int t_boundary = 1)
{
    // constexpr int kWarpLines = 8;
    // constexpr int


    using Argument = typename WilsonDslashDevice<FloatType_, BlockShape_, WarpShape_, Nspin_, TensorOpEnabled_>::Argument;

    assert(BlockShape_::kM > 0 && BlockShape_::kN > 0 && BlockShape_::kK > 0);

    // z 轴切分矩阵坐标点，(x,y)切分单个矩阵
    int block_id = blockIdx.z;
    int grid_size = gridDim.z;  // 1D grid
    int half_vol = latt_desc.half_lattice_volume();

    WilsonDslashDevice<FloatType_, BlockShape_, WarpShape_, Nspin_, TensorOpEnabled_> wilson_op(n_color, m_rhs);
    for (int i = block_id; i < half_vol; i += grid_size) {
        Argument arg {
            .dagger_flag = dagger_flag,
            .out_half = out_half,
            .in_half = in_half,   // 计算out-even的时候，in为odd
            .gauge = gauge,
            .latt_desc = latt_desc,
            .multiprocess = multiprocess,
            .parity = parity,
            .n_color = n_color,
            .m_rhs = m_rhs,
            .coord_1dim = i,
            .kappa = kappa,
            .t_boundary = t_boundary
        };

        wilson_op(arg, mat_flag);
    }
}


}

