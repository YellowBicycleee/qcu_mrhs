//
// Created by wangj on 2024/12/9.

#ifndef SUN_MRHS_WILSON_DSLASH_SIMT_GHOST_CUH
#define SUN_MRHS_WILSON_DSLASH_SIMT_GHOST_CUH

#include "qcu_helper.h"
#include "qcu_utils.h"
namespace qcu::device {
template <
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8>,
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>,
    int Stages = 1,
    typename Float2 = Float2_t<FloatType_>,
    typename Complex = Complex<FloatType_>
>
QCU_DEVICE
void single_point_wilson_dslash_t_forward_ghost_pack(
    FloatType_* __restrict__ out,
    FloatType_* __restrict__ in,
    FloatType_* __restrict__ gauge,
    QcuLattDesc latt_desc, int ghost_dim , int parity,
    bool dagger_flag, int n_color, int m_rhs, int coord_1dim)
{
    using GaugeMatShape = gemm::MatShape<BlockShape_::kM, BlockShape_::kK>;
    using FermionMatShape = gemm::MatShape<BlockShape_::kK, BlockShape_::kN>;
    constexpr int A_Shape = GaugeMatShape::kMN;
    constexpr int B_Shape = FermionMatShape::kMN;
    constexpr int dir = BWD; // send to forward, but from unpack process, it is from backward

    if (ghost_dim < 0 || ghost_dim >= Nd) {
        fprintf(stdout, "Error: ghost_dim is out of range\n");
        cuda_abort();
    }

    const int fermion_site_length = n_color * m_rhs;

    __shared__ Float2 smem_A[Stages][A_Shape];
    __shared__ Float2 smem_B[Stages][B_Shape * 2];

    // ldg_A and ldg_B are used to load A and B from global memory
    Complex ldg_A[1];
    Complex ldg_B[1];
    // store temp_res into register, then stg to global memory
    Complex temp_res[2][1];

    // 4-dim lattice desc
    QcuLattDesc latt_half_desc{latt_desc.X() >> 1, latt_desc.Y(), latt_desc.Z(), latt_desc.T()};
    // 3-dim sub-space lattice desc
    QcuLattDesc sub_space_half_desc {latt_half_desc};
    if (ghost_dim > 0 && ghost_dim < Nd) {
        sub_space_half_desc.at(ghost_dim) = 1; // a 3-dim desc hyperplane of 4 dim space
    }
    else {
        fprintf(stdout, "X dim not implemented yet\n");
    }

    Point sub_latt_coord {
        coord_1dim % sub_space_half_desc.X()
        , coord_1dim % (sub_space_half_desc.Y() * sub_space_half_desc.X()) / sub_space_half_desc.X()
        , coord_1dim % (sub_space_half_desc.Z() * sub_space_half_desc.Y() * sub_space_half_desc.X()) / (sub_space_half_desc.Y() * sub_space_half_desc.X())
        , coord_1dim / (sub_space_half_desc.Z() * sub_space_half_desc.Y() * sub_space_half_desc.X())
        , parity
    };

    Point coord {sub_latt_coord};
    // send to forward
    if (dir == BWD) {  coord.at(ghost_dim) = latt_half_desc.at(ghost_dim) - 1; }

    int32_t mat1_pos; // will be 0 or 1, use this to set mat1 position
    int32_t mat2_pos; // will be 2 or 3, use this to set mat2 position     temp_mat = mat1 + scale * mat2

    int32_t blocks_m = div_ceil(n_color, BlockShape_::kM);
    int32_t blocks_n = div_ceil(m_rhs, BlockShape_::kN);

    Complex scale; // when read B, use B1 + scale B2

    for (int loop_blk_m = blockIdx.y; loop_blk_m < blocks_m; loop_blk_m += gridDim.y) {
        for (int loop_blk_n = blockIdx.x; loop_blk_n < blocks_n; loop_blk_n += gridDim.x) {

            int row = loop_blk_m * BlockShape_::kM;
            int col = loop_blk_n * BlockShape_::kN;

            // calculate start addr of global A and B
            // FWD in pack, BWD in unpack
            Float2* glb_A = reinterpret_cast<Float2 *>(coord.getGaugeAddr(gauge, ghost_dim, latt_half_desc, n_color));
            Float2* glb_B = reinterpret_cast<Float2 *>(coord.getGatheredColorSpinorAddr(in, latt_half_desc, n_color, m_rhs));

            // main loop
            for (int k = 0; k < n_color; k += BlockShape_::kK) {
                /// load Gauge
                // global memory is col-major, col-major in smem
                gemm::ldg<Float2, gemm::MatShapeTranspose<GaugeMatShape>, BlockShape_, WarpShape_>
                    (glb_A, n_color, n_color, k, row, reinterpret_cast<Float2*>(ldg_A));
                // dagger
                for (int i = 0; i < sizeof(ldg_A) / sizeof(Complex); i++) { ldg_A[i] = ldg_A[i].conj(); }
                gemm::sts_transpose<Float2, GaugeMatShape, BlockShape_, WarpShape_> (smem_A[0], reinterpret_cast<Float2*>(ldg_A));
                __syncthreads();

                /// load Fermion
                /// load B from global memory to register, need combine 2 of 4 in global memory to 2 in smem
                #pragma unroll
                for (int pos = 0; pos < 2; ++pos) {
                    if (row < n_color && col < m_rhs) {
                        mat1_pos = pos;
                        mat2_pos = kernel::Gamma<FloatType_>::get_reconstruct_mat_id(ghost_dim, mat1_pos);
                        // get scale
                        scale = kernel::Gamma<FloatType_>::get_projection_scale(ghost_dim, mat1_pos, dir);
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
            } // end main loop for

            // store res to global memory
            // the address calculated from hyperplane coord
            Float2* glb_out = reinterpret_cast<Float2 *>(sub_latt_coord.getGatheredColorSpinorAddr(out, sub_space_half_desc, n_color, m_rhs));

            // epilogue, store into global memory
            #pragma unroll
            for (int i = 0; i < Nd / 2; ++i) {
                gemm::stg<Float2, FermionMatShape, BlockShape_, WarpShape_> (
                    reinterpret_cast<Float2*>(glb_out) + i * n_color * m_rhs,
                    n_color, m_rhs, row, col,
                    reinterpret_cast<Float2*>(temp_res[i]));
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
    typename WarpShape_ = gemm::GemmShape<8, 8, 4>
>
QCU_GLOBAL
void wilson_dslash_su_n_mrhs_ghost(
    FloatType_* __restrict__ out,
    FloatType_* __restrict__ in,
    FloatType_* __restrict__ gauge,
    QcuLattDesc latt_desc, int ghost_dim,
    int parity, bool dagger_flag, int n_color, int m_rhs)
{
    assert(BlockShape_::kM > 0 && BlockShape_::kN > 0 && BlockShape_::kK > 0);

    // z 轴切分矩阵坐标点，(x,y)切分单个矩阵
    int block_id = blockIdx.z;
    int grid_size = gridDim.z;  // 1D grid
    int half_vol = latt_desc.half_lattice_volume();

} // function wilson_dslash_su_n_mrhs_ghost


} // namespace qcu::device
#endif //SUN_MRHS_WILSON_DSLASH_SIMT_GHOST_CUH
