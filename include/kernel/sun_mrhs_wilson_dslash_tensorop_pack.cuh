#pragma once
#include "qcu_helper.h"
#include "kernel/gemm/qcu_gemm_configure.cuh"
#include "kernel/gemm/qcu_gemm_loader.cuh"
#include "kernel/qcu_gamma.cuh"
#include "point/qcu_point.cuh"
#include "qcu_utils.h"
#include "complex/qcu_complex.cuh"
#include "kernel/sun_mrhs_wilson_dslash_load_gauge_fermion.cuh"
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
class WilsonDslashDevicePack {
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

    QCU_DEVICE void backward_pack(Argument& arg, int ghost_dim) {
        constexpr int kElemsPerThread = WarpShape_::kMN / kWarpSize;
        constexpr int dir = FWD; // send to backward, but from unpack process, it is from forward
        if (ghost_dim < 0 || ghost_dim >= Nd) {
            printf("Error: ghost_dim is out of range\n");
            cuda_abort();
        }
        const int fermion_site_length = arg.n_color * arg.m_rhs;

        int warp_rank = (threadIdx.y * blockDim.x + threadIdx.x) / kWarpSize;
        int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % kWarpSize;
        int warp_rank_row = warp_rank / kWarpNumCol;
        int warp_rank_col = warp_rank % kWarpNumCol;

        // 4-dim lattice desc
        QcuLattDesc latt_half_desc{arg.latt_desc.X() >> 1, arg.latt_desc.Y(), arg.latt_desc.Z(), arg.latt_desc.T()};

        // 3-dim sub-space lattice desc
        QcuLattDesc sub_space_half_desc;
        if (ghost_dim == X_DIM) {
            sub_space_half_desc = QcuLattDesc{1, arg.latt_desc.Y() >> 1, arg.latt_desc.Z(), arg.latt_desc.T()};
        }
        else {
            sub_space_half_desc = QcuLattDesc {latt_half_desc};
            sub_space_half_desc.at(ghost_dim) = 1; // a 3-dim desc hyperplane of 4 dim space
        }

        Point<Nspin_> sub_latt_coord {
            arg.coord_1dim % sub_space_half_desc.X()
            , arg.coord_1dim % (sub_space_half_desc.Y() * sub_space_half_desc.X()) / sub_space_half_desc.X()
            , arg.coord_1dim % (sub_space_half_desc.Z() * sub_space_half_desc.Y() * sub_space_half_desc.X()) / (sub_space_half_desc.Y() * sub_space_half_desc.X())
            , arg.coord_1dim / (sub_space_half_desc.Z() * sub_space_half_desc.Y() * sub_space_half_desc.X())
            , arg.parity
        };

        Point<Nspin_> coord {sub_latt_coord}; // coord of whole lattice
        if (ghost_dim == X_DIM) {
            int cb_xzt = (sub_latt_coord.Z() + sub_latt_coord.T()) % 2;
            coord.at(Y_DIM) = 2 * sub_latt_coord.Y() + (cb_xzt == arg.parity);
        }
        coord.at(ghost_dim) = 0;
        coord.setParity(1 - arg.parity);

        int mat1_pos;
        int mat2_pos;

        int blocks_m = div_ceil(arg.n_color, BlockShape_::kM);
        int blocks_n = div_ceil(arg.m_rhs, BlockShape_::kN);
        int groupId = (lane_id >> 2);
        int threadID_in_group = lane_id % 4;
        Complex_ scale; // when read B, use B1 + scale B2
        // calculate start addr of global and B
        Float2_* glb_B = reinterpret_cast<Float2_ *>(coord.getGatheredColorSpinorAddr(arg.in_half, latt_half_desc, arg.n_color, arg.m_rhs));
        Float2_* glb_out = reinterpret_cast<Float2_ *>(sub_latt_coord.getGatheredHalfColorSpinorAddr(arg.out_half, sub_space_half_desc, arg.n_color, arg.m_rhs));
        int warp_row_offset = warp_rank_row * WarpShape_::kM;
        int warp_col_offset = warp_rank_col * WarpShape_::kN;

        for (int loop_blk_m = blockIdx.y; loop_blk_m < blocks_m; loop_blk_m += gridDim.y) {
            for (int loop_blk_n = blockIdx.x; loop_blk_n < blocks_n; loop_blk_n += gridDim.x) {

                int block_row = loop_blk_m * BlockShape_::kM;
                int block_col = loop_blk_n * BlockShape_::kN;

                // projection scale
                for (mat1_pos = 0; mat1_pos < 2; mat1_pos++) {
                    mat2_pos = kernel::Gamma<Float_>::get_reconstruct_mat_id(ghost_dim, mat1_pos);
                    // get scale
                    scale = kernel::Gamma<Float_>::get_projection_scale(ghost_dim, mat1_pos, dir);
                    if (arg.dagger_flag) { scale = -scale; }

                    // load and store
                    Float2_* start = glb_out + mat1_pos * fermion_site_length;

                    int row, col;
                    for (int idx = 0; idx < kElemsPerThread; ++idx) {
                        if (idx < 2 || (idx >= 4 && idx < 6)) { row = groupId; }
                        else { row = groupId + 8; }
                        if (idx < 4) { col = (threadID_in_group * 2) + (idx & 0x1); }
                        else { col = (threadID_in_group * 2) + (idx & 0x1) + 8; }

                        int row_in_global = block_row + warp_row_offset + row;
                        int col_in_global = block_col + warp_col_offset + col;

                        if (row_in_global < arg.n_color && col_in_global < arg.m_rhs) {
                            Complex_ temp1 = Complex_(glb_B[mat1_pos * fermion_site_length + row_in_global * arg.m_rhs + col_in_global]);
                            Complex_ temp2 = Complex_(glb_B[mat2_pos * fermion_site_length + row_in_global * arg.m_rhs + col_in_global]);
                            Complex_ res = temp1 + scale * temp2;
                            start[row_in_global * arg.m_rhs + col_in_global] = *reinterpret_cast<Float2_*>(&res);
                        }
                    }
                }
            }
        }
    }

    QCU_DEVICE void forward_pack(Argument& arg, int ghost_dim) {
        constexpr int kElemsPerThread = WarpShape_::kMN / kWarpSize;
        constexpr int dir = BWD; // send to forward, but from unpack process, it is from backward
        if (ghost_dim < 0 || ghost_dim >= Nd) {
            printf("Error: ghost_dim is out of range\n");
            cuda_abort();
        }
        const int fermion_site_length = arg.n_color * arg.m_rhs;

        __shared__ Float_ A_tile_real[BlockShape_::kMK];
        __shared__ Float_ A_tile_imag[BlockShape_::kMK];
        __shared__ Float_ B_tile_real[2][BlockShape_::kKN]; // Ns / 2
        __shared__ Float_ B_tile_imag[2][BlockShape_::kKN];

        Complex_ temp_result[2][kElemsPerThread];  // Nspin / 2
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < kElemsPerThread; j++) {
                temp_result[i][j] = Complex_(0, 0);
            }
        }

        // 4-dim lattice desc
        QcuLattDesc latt_half_desc{arg.latt_desc.X() >> 1, arg.latt_desc.Y(), arg.latt_desc.Z(), arg.latt_desc.T()};

        // 3-dim sub-space lattice desc
        QcuLattDesc sub_space_half_desc;
        if (ghost_dim == X_DIM) {
            sub_space_half_desc = QcuLattDesc{1, arg.latt_desc.Y() >> 1, arg.latt_desc.Z(), arg.latt_desc.T()};
        }
        else {
            sub_space_half_desc = QcuLattDesc {latt_half_desc};
            sub_space_half_desc.at(ghost_dim) = 1; // a 3-dim desc hyperplane of 4 dim space
        }

        Point<Nspin_> sub_latt_coord {
            arg.coord_1dim % sub_space_half_desc.X()
            , arg.coord_1dim % (sub_space_half_desc.Y() * sub_space_half_desc.X()) / sub_space_half_desc.X()
            , arg.coord_1dim % (sub_space_half_desc.Z() * sub_space_half_desc.Y() * sub_space_half_desc.X()) / (sub_space_half_desc.Y() * sub_space_half_desc.X())
            , arg.coord_1dim / (sub_space_half_desc.Z() * sub_space_half_desc.Y() * sub_space_half_desc.X())
            , arg.parity
        };

        Point<Nspin_> coord {sub_latt_coord}; // coord of whole lattice
        if (ghost_dim == X_DIM) {
            int cb_xzt = 1 - (sub_latt_coord.Z() + sub_latt_coord.T()) % 2;
            coord.at(Y_DIM) = 2 * sub_latt_coord.Y() + (cb_xzt == arg.parity);
        }
        coord.at(ghost_dim) = latt_half_desc.at(ghost_dim) - 1;
        coord.setParity(1 - arg.parity);

        int32_t mat1_pos;
        int32_t mat2_pos;

        int blocks_m = div_ceil(arg.n_color, BlockShape_::kM);
        int blocks_n = div_ceil(arg.m_rhs, BlockShape_::kN);
        int warp_rank = (threadIdx.y * blockDim.x + threadIdx.x) / kWarpSize;
        int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % kWarpSize;
        int warp_rank_row = warp_rank / kWarpNumCol;
        int warp_rank_col = warp_rank % kWarpNumCol;
        int warp_row_offset = warp_rank_row * WarpShape_::kM;
        int warp_col_offset = warp_rank_col * WarpShape_::kN;
        int groupId = (lane_id >> 2);
        int threadID_in_group = lane_id % 4;
        Complex_ scale; // when read B, use B1 + scale B2

        // calculate start addr of global A and B
        // FWD in pack, BWD in unpack
        Float2_* glb_A = reinterpret_cast<Float2_ *>(coord.getGaugeAddr(arg.gauge, ghost_dim, latt_half_desc, arg.n_color));
        Float2_* glb_B = reinterpret_cast<Float2_ *>(coord.getGatheredColorSpinorAddr(arg.in_half, latt_half_desc, arg.n_color, arg.m_rhs));
        Float2_* glb_out = reinterpret_cast<Float2_ *>(sub_latt_coord.getGatheredHalfColorSpinorAddr(arg.out_half, sub_space_half_desc, arg.n_color, arg.m_rhs));

        for (int loop_blk_m = blockIdx.y; loop_blk_m < blocks_m; loop_blk_m += gridDim.y) {
            for (int loop_blk_n = blockIdx.x; loop_blk_n < blocks_n; loop_blk_n += gridDim.x) {

                int block_row = loop_blk_m * BlockShape_::kM;
                int block_col = loop_blk_n * BlockShape_::kN;

                wmma::fragment<wmma::accumulator, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_> temp_r[2];
                wmma::fragment<wmma::accumulator, WarpShape_::kM, WarpShape_::kN, WarpShape_::kK, Float_> temp_i[2];
                for (int pos = 0; pos < 2; ++pos) {
                    wmma::fill_fragment(temp_r[pos], 0.0f);
                    wmma::fill_fragment(temp_i[pos], 0.0f);
                }

                // main loop
                for (int k = 0; k < arg.n_color; k += BlockShape_::kK) {
                    /// load Gauge
                    ldg_and_dagger_to_sts<Float_, GaugeMatShape>(glb_A, k, block_row, arg.n_color, arg.n_color, A_tile_real, A_tile_imag);

                    // ldg Fermion
                    #pragma unroll
                    for (int pos = 0; pos < 2; ++pos) {
                        if (block_row < arg.n_color && block_col < arg.m_rhs) {
                            mat1_pos = pos;
                            mat2_pos = kernel::Gamma<Float_>::get_reconstruct_mat_id(ghost_dim, mat1_pos);
                            // get scale
                            scale = kernel::Gamma<Float_>::get_projection_scale(ghost_dim, mat1_pos, dir);
                            if (arg.dagger_flag) { scale = -scale; }
                        }

                        ldg_fermion_sts<Float_, FermionMatShape>(
                            glb_B + mat1_pos * fermion_site_length,
                            glb_B + mat2_pos * fermion_site_length,
                            k, block_col, arg.n_color, arg.m_rhs,
                            scale, &(B_tile_real[pos][0]), &(B_tile_imag[pos][0])
                            );
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
                } // end main loop for

                // add to result
                if (block_row < arg.n_color && block_col < arg.m_rhs) {
                    for (mat1_pos = 0; mat1_pos < 2; ++mat1_pos) {
                        for (int elem_idx = 0; elem_idx < kElemsPerThread; ++elem_idx) {
                            Complex_ temp_res(temp_r[mat1_pos].x[elem_idx], temp_i[mat1_pos].x[elem_idx]);
                            temp_result[mat1_pos][elem_idx] += temp_res;
                        }
                    }
                }

                #pragma unroll
                // store thread result to global memory
                for (int mat1_pos = 0; mat1_pos < Nspin_ / 2; ++mat1_pos) { // store global memory
                    Float2_* start = reinterpret_cast<Float2_*>(glb_out) + mat1_pos * fermion_site_length;

                    int row, col;
                    for (int idx = 0; idx < kElemsPerThread; ++idx) {
                        if (idx < 2 || (idx >= 4 && idx < 6)) {
                            row = groupId;
                        } else {
                            row = groupId + 8;
                        }
                        if (idx < 4) {
                            col = (threadID_in_group * 2) + (idx & 0x1);
                        }
                        else {
                            col = (threadID_in_group * 2) + (idx & 0x1) + 8;
                        }
                        int row_in_global = block_row + warp_row_offset + row;
                        int col_in_global = block_col + warp_col_offset + col;
                        if (row_in_global < arg.n_color && col_in_global < arg.m_rhs) {
                            start[row_in_global * arg.m_rhs + col_in_global]
                                    = reinterpret_cast<Float2_*>(&(temp_result[mat1_pos][0]))[idx];
                        }
                    }
                }
            }
        }
    }

    QCU_DEVICE WilsonDslashDevicePack (int n_color, int m_rhs)
        : blocks_m (div_ceil(n_color, BlockShape_::kM))
        , blocks_n (div_ceil(m_rhs, BlockShape_::kN))
    {}

private:
    using GaugeMatShape = gemm::MatShape<BlockShape_::kM, BlockShape_::kK>;
    using FermionMatShape = gemm::MatShape<BlockShape_::kK, BlockShape_::kN>;
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
void wilson_dslash_sun_mrhs_forward_ghost_pack(
    FloatType_* __restrict__ out_half,
    FloatType_* __restrict__ in_half,
    FloatType_* __restrict__ gauge,
    int ghost_dim,
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

    using Argument = typename WilsonDslashDevicePack<FloatType_, BlockShape_, WarpShape_, Nspin_, TensorOpEnabled_>::Argument;

    assert(BlockShape_::kM > 0 && BlockShape_::kN > 0 && BlockShape_::kK > 0);

    // z 轴切分矩阵坐标点，(x,y)切分单个矩阵
    int block_id = blockIdx.z;
    int grid_size = gridDim.z;  // 1D grid
    int half_vol = latt_desc.half_lattice_volume();

    WilsonDslashDevicePack<FloatType_, BlockShape_, WarpShape_, Nspin_, TensorOpEnabled_> wilson_op(n_color, m_rhs);
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

        wilson_op.forward_pack(arg, ghost_dim);
    }
}

template <
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    int Nspin_ = 4,
    bool TensorOpEnabled_ = true
>
QCU_GLOBAL
void wilson_dslash_sun_mrhs_backward_ghost_pack(
    FloatType_* __restrict__ temp_out_half,
    FloatType_* __restrict__ in_half,
    FloatType_* __restrict__ gauge,
    int ghost_dim,
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
    using Argument = typename WilsonDslashDevicePack<FloatType_, BlockShape_, WarpShape_, Nspin_, TensorOpEnabled_>::Argument;

    assert(BlockShape_::kM > 0 && BlockShape_::kN > 0 && BlockShape_::kK > 0);

    // z 轴切分矩阵坐标点，(x,y)切分单个矩阵
    int block_id = blockIdx.z;
    int grid_size = gridDim.z;  // 1D grid
    int half_vol = latt_desc.half_lattice_volume();

    WilsonDslashDevicePack<FloatType_, BlockShape_, WarpShape_, Nspin_, TensorOpEnabled_> wilson_op(n_color, m_rhs);
    for (int i = block_id; i < half_vol; i += grid_size) {
        Argument arg {
            .dagger_flag = dagger_flag,
            .out_half = temp_out_half,
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
        wilson_op.backward_pack(arg, ghost_dim);
    }
}

}

