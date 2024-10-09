#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#include <cassert>

#include "kernel/complex_matmul.cuh"
#include "qcu_public.h"
#include "base/datatype/qcu_float2.cuh"
#include "qcu_utils.h"
#include "qcu_wmma_constant.h"
namespace qcu {
namespace device {

using namespace nvcuda;

// clang-format off
/** 
 * @name x_dim_fwd_dir
 * @brief x dimension, forward direction
 * @tparam Float data type
 * @param smem_mat_L shared memory for L
 * @param smem_mat_U shared memory for U
 * @param smem_mat_R shared memory for R
 * @param smem_mat_T shared memory for T
 * @param global_gauge global memory for gauge
 * @param global_fermion_in global memory for fermion
 * @param dagger_flag dagger flag
 * @param n_color number of color
 * @param m_rhs number of rhs
 * @details [R1, R2] = U[T1, T2]      L1 = L1 + R1, L2 = L2 + R2, L3 = L3 + iR2, L4 = L4 + iR1
 *          smems are chunks separated by warp, calculated over caller function
 *          Matrix M donnot have to store in smem, just load from global memory to register, combine to T1, T2 elem then store T1 and T2
 *          smem STORE: T[1, 2] (combination of in)   R[1, 2](temp result)   L[1, 2](smem out)    U(gauge)
 *          U : real: smem_mat_U                           imag: smem_mat_U + WMMA_M * WMMA_K
 *          T : real: smem_mat_T                           imag: smem_mat_T + WMMA_M * WMMA_N
 *          R : real: smem_mat_R                           imag: smem_mat_R + WMMA_M * WMMA_N
*/

template <typename Float>
[[deprecated("Use function dslash_mat_mul_new instead")]]
__device__ void dslash_mat_mul(Float* __restrict__ smem_mat_L, Float* __restrict__ smem_mat_U,
                               Float* __restrict__ smem_mat_R, Float* __restrict__ smem_mat_T,
                               Float* __restrict__ global_gauge, Float* __restrict__ global_fermion_in,
                               bool dagger_flag, int n_color, int m_rhs, int warp_begin_row, int warp_begin_col,
                               int dim, int dir) {
    // clang-format on
    using Float2 = typename Float2Wrapper<Float>::Float2;
    constexpr int WMMA_M = WMMA_Param<Float>::WMMA_M;
    constexpr int WMMA_N = WMMA_Param<Float>::WMMA_N;
    constexpr int WMMA_K = WMMA_Param<Float>::WMMA_K;

    int k_tiles = div_ceil(n_color, WMMA_K);

    Float* smem_U = smem_mat_U;
    Float* smem_T = smem_mat_T;
    Float* smem_R = smem_mat_R;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> R1_real_frag;  // accumulator C_frag
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> R1_imag_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> R2_real_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> R2_imag_frag;

    wmma::fill_fragment(R1_real_frag, 0.0f);  // zero fitting R1_real_frag
    wmma::fill_fragment(R1_imag_frag, 0.0f);  // zero fitting R1_imag_frag
    wmma::fill_fragment(R2_real_frag, 0.0f);  // zero fitting R2_real_frag
    wmma::fill_fragment(R2_imag_frag, 0.0f);  // zero fitting R2_imag_frag

    bool if_dagger_u;
    bool local_dagger_flag;
    int gamma_idx = dim + 1;  // gamma{1,2,3,4} ----- gamma{X_DIM + 1, Y_DIM + 1, Z_DIM + 1, T_DIM + 1}
    if (dir == FWD) {
        if_dagger_u = false;
        local_dagger_flag = dagger_flag;
    } else if (dir == BWD) {
        if_dagger_u = true;
        local_dagger_flag = !dagger_flag;
    } else {
        assert(0);
    }

    for (int i = 0; i < k_tiles; i++) {
        // load U
        // global_iter_start_m = warp_begin_row, global_iter_start_n = i * WMMA_K
        load_complex_gauge_mat_from_global_to_smem(smem_U, WMMA_M, WMMA_K, global_gauge, warp_begin_row, i * WMMA_K,
                                                   if_dagger_u, n_color);

        // load T1,  T1 = M1 - iM4 = (M1.real + M4.imag) + i(M1.imag - M4.real)
        // load T1
        // global_iter_start_k = i * WMMA_K,      global_iter_start_n = warp_begin_col
        load_complex_fermion_mat_T1_from_global_to_smem(smem_T, WMMA_K, WMMA_N, global_fermion_in, i * WMMA_K,
                                                        warp_begin_col, gamma_idx, local_dagger_flag, n_color, m_rhs);

        // matmul  R1 = U T1
        tensor_core_complex_matmul(smem_U, smem_T, R1_real_frag, R1_imag_frag);

        // load T2
        // global_iter_start_k = i * WMMA_K,      global_iter_start_n = warp_begin_col
        load_complex_fermion_mat_T2_from_global_to_smem(smem_T, WMMA_K, WMMA_N, global_fermion_in, i * WMMA_K,
                                                        warp_begin_col, gamma_idx, local_dagger_flag, n_color, m_rhs);
        // matmul  R2 = U T2
        tensor_core_complex_matmul(smem_U, smem_T, R2_real_frag, R2_imag_frag);

    }  // for i : K_TILES
    // store to smem_R
    wmma::store_matrix_sync(smem_R + 0 * WMMA_M * WMMA_N, R1_real_frag, WMMA_N, wmma::mem_row_major);  // R1_real
    wmma::store_matrix_sync(smem_R + 1 * WMMA_M * WMMA_N, R1_imag_frag, WMMA_N, wmma::mem_row_major);  // R1_imag
    wmma::store_matrix_sync(smem_R + 2 * WMMA_M * WMMA_N, R2_real_frag, WMMA_N, wmma::mem_row_major);  // R2_real
    wmma::store_matrix_sync(smem_R + 3 * WMMA_M * WMMA_N, R2_imag_frag, WMMA_N, wmma::mem_row_major);  // R2_imag
    // acc to smem_L    TODO
    calc_L_from_R1(smem_mat_L, smem_R, WMMA_M, WMMA_N, gamma_idx, local_dagger_flag);
    calc_L_from_R2(smem_mat_L, smem_R, WMMA_M, WMMA_N, gamma_idx, local_dagger_flag);
}


template <typename Float>
__device__ void dslash_mat_mul_new(
    wmma::fragment<wmma::accumulator, WMMA_Param<Float>::WMMA_M, 
                                      WMMA_Param<Float>::WMMA_N, 
                                      WMMA_Param<Float>::WMMA_K,
                                    Float>* L_frag_ptr,
    wmma::fragment<wmma::accumulator, WMMA_Param<Float>::WMMA_M, 
                                      WMMA_Param<Float>::WMMA_N, 
                                      WMMA_Param<Float>::WMMA_K,
                                    Float>* R_frag_ptr,
    Float* __restrict__ smem_mat_U, Float* __restrict__ smem_mat_R, Float* __restrict__ smem_mat_T,
    Float* __restrict__ global_gauge, Float* __restrict__ global_fermion_in, bool dagger_flag, int n_color, int m_rhs,
    int warp_begin_row, int warp_begin_col, int dim, int dir) {
    // clang-format on
    using Float2 = typename Float2Wrapper<Float>::Float2;
    constexpr int WMMA_M = WMMA_Param<Float>::WMMA_M;
    constexpr int WMMA_N = WMMA_Param<Float>::WMMA_N;
    constexpr int WMMA_K = WMMA_Param<Float>::WMMA_K;

    int k_tiles = div_ceil(n_color, WMMA_K);

    Float* smem_U = smem_mat_U;
    Float* smem_T = smem_mat_T;

    // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> R_frag[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::fill_fragment(R_frag_ptr[i], 0.0f);
    }

    bool if_dagger_u;
    bool local_dagger_flag;
    int gamma_idx = dim + 1;  // gamma{1,2,3,4} ----- gamma{X_DIM + 1, Y_DIM + 1, Z_DIM + 1, T_DIM + 1}
    if (dir == FWD) {
        if_dagger_u = false;
        local_dagger_flag = dagger_flag;
    } else if (dir == BWD) {
        if_dagger_u = true;
        local_dagger_flag = !dagger_flag;
    } else {
        assert(0);
    }

    for (int i = 0; i < k_tiles; i++) {
        // load U
        // global_iter_start_m = warp_begin_row, global_iter_start_n = i * WMMA_K
        load_complex_gauge_mat_from_global_to_smem(smem_U, WMMA_M, WMMA_K, global_gauge, warp_begin_row, i * WMMA_K,
                                                   if_dagger_u, n_color);

        // load T1,  T1 = M1 - iM4 = (M1.real + M4.imag) + i(M1.imag - M4.real)
        // load T1
        // global_iter_start_k = i * WMMA_K,      global_iter_start_n = warp_begin_col
        load_complex_fermion_mat_T1_from_global_to_smem(smem_T, WMMA_K, WMMA_N, global_fermion_in, i * WMMA_K,
                                                        warp_begin_col, gamma_idx, local_dagger_flag, n_color, m_rhs);

        // matmul  R1 = U T1
        tensor_core_complex_matmul(smem_U, smem_T, R_frag_ptr[0], R_frag_ptr[1]);  // R1_real, R1_imag

        // load T2
        // global_iter_start_k = i * WMMA_K,      global_iter_start_n = warp_begin_col
        load_complex_fermion_mat_T2_from_global_to_smem(smem_T, WMMA_K, WMMA_N, global_fermion_in, i * WMMA_K,
                                                        warp_begin_col, gamma_idx, local_dagger_flag, n_color, m_rhs);
        // matmul  R2 = U T2
        tensor_core_complex_matmul(smem_U, smem_T, R_frag_ptr[2], R_frag_ptr[3]);  // R2_real, R2_imag

    }  // for i : K_TILES

    // acc to smem_L
    calc_L_from_R<Float>(L_frag_ptr, R_frag_ptr, gamma_idx, local_dagger_flag);
}

}  // namespace device
}  // namespace qcu