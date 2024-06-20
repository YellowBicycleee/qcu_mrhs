#pragma once

#include <cuda_fp16.h>

#include "kernel/constants.cuh"
#include "qcu_float_float2_wrapper.h"
#include "qcu_utils.h"
#include "qcu_wmma_constant.h"

namespace qcu {
namespace device {

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
 *          T1: real: smem_mat_T                           imag: smem_mat_T + WMMA_M * WMMA_N
 *          T2: real: smem_mat_T + 2 * WMMA_M * WMMA_N     imag: smem_mat_T + 3 * WMMA_M * WMMA_N
 *          R1: real: smem_mat_R                           imag: smem_mat_R + WMMA_M * WMMA_N
 *          R2: real: smem_mat_R + 2 * WMMA_M * WMMA_N     imag: smem_mat_R + 3 * WMMA_M * WMMA_N
 *          U:  real: smem_mat_U                           imag: smem_mat_U + WMMA_M * WMMA_K
*/
template <typename Float>
__device__ void x_dim_fwd_dir(Float* __restrict__ smem_mat_L, Float * __restrict__ smem_mat_U,
                              Float2Wrapper<Float>::AccFloat* __restrict__ smem_mat_R, 
                              Float2Wrapper<Float>::AccFloat* __restrict__ smem_mat_T,
                              Float* __restrict__ global_gauge, Float* __restrict__ global_fermion_in, 
                              int dagger_flag, int n_color, int m_rhs) {
    // clang-format on

    using Float2 = Float2Wrapper<Float>::Float2;
    using AccFloat = Float2Wrapper<Float>::AccFloat;

    constexpr int WMMA_M = WMMA_Param<Float>::WMMA_M;
    constexpr int WMMA_N = WMMA_Param<Float>::WMMA_N;
    constexpr int WMMA_K = WMMA_Param<Float>::WMMA_K;

    // complex warp row
    // complex warp col
    int warp_cnt = blockDim.y;
    int warp_id = threadIdx.y;
    int total_warps_row = div_ceil(n_color, WMMA_M);
    int total_warps_col = div_ceil(m_rhs, WMMA_N);

    int k_tiles = div_ceil(n_color, WMMA_K);
    // m * n / (WWMA_M * WMMA_N)
    // int virtual_warp_num = (n_color * m_rhs) / (WMMA_M * WMMA_N);

    for (int virtual_warp_id = warp_id; i < total_warps_row * total_warps_col; virtual_warp_id += warp_cnt) {
        int warp_id_i = virtual_warp_id / total_warps_col;
        int warp_id_j = virtual_warp_id % total_warps_col;

        int warp_begin_row = warp_id_i * WMMA_M;
        int warp_begin_col = warp_id_j * WMMA_N;

        Float2 temp1;
        Float2 temp2;

        Float* smem_u_real = smem_mat_U;
        Float* smem_u_imag = smem_mat_U + WMMA_M * WMMA_K;

        AccFloat* smem_t_real = smem_mat_T;
        AccFloat* smem_t_imag = smem_mat_T + WMMA_M * WMMA_N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> C_frag;
        wmma::fill_fragment(C_frag, 0.0f);  // zero fitting C_frag
        Float* smem_r_real;
        Float* smem_r_imag;
        // clear smem_R
        // R1
        smem_r_real = smem_mat_R;
        smem_r_imag = smem_mat_R + WMMA_M * WMMA_N;
        for (int i = threadIdx.x; i < WMMA_M * WMMA_N; i += WARP_SIZE) {
            smem_r_real[i] = 0;
            smem_r_imag[i] = 0;
        }
        __syncwarp();
        // R2
        smem_r_real = smem_mat_R + 2 * WMMA_M * WMMA_N;
        smem_r_imag = smem_mat_R + 3 * WMMA_M * WMMA_N;
        for (int i = threadIdx.x; i < WMMA_M * WMMA_N; i += WARP_SIZE) {
            smem_r_real[i] = 0;
            smem_r_imag[i] = 0;
        }
        __syncwarp();
        for (int i = 0; i < k_tiles; i++) {
            Float2* loader_ptr1;
            Float2* loader_ptr2;

            int local_i;
            int local_j;
            int global_i;
            int global_j;

            // load U
            for (int j = threadIdx.x; j < WMMA_M * WMMA_K; j += WARP_SIZE) {
                loader_ptr1 = reinterpret_cast<Float2*>(global_gauge);
                local_i = j / WMMA_K;
                local_j = j % WMMA_K;
                global_i = warp_begin_row + local_i;
                global_j = i * WMMA_K + local_j;
                if (global i < n_color && global_j < n_color) {
                    temp1 = loader_ptr1[IDX2D(global_i, global_j, n_color)];
                } else {  // zero padding
                    temp1.x = 0;
                    temp1.y = 0;
                }

                smem_u_real[j] = temp1.x;
                smsm_u_imag[j] = temp1.y;
            }
            __syncwarp();

            // load T1,  T1 = M1 - iM4 = (M1.real + M4.imag) + i(M1.imag - M4.real)
            smem_t_real = smem_mat_T;
            smem_t_imag = smem_mat_T + WMMA_M * WMMA_N;
            for (int j = threadIdx.x; j < WMMA_K * WMMA_N; j += WARP_SIZE) {
                loader_ptr1 = reinterpret_cast<Float2*>(global_fermion_in);
                loader_ptr2 = reinterpret_cast<Float2*>(global_fermion_in) + 3 * n_color * m_rhs;
                local_i = j / WMMA_N;
                local_j = j % WMMA_N;
                global_i = i * WMMA_K + local_i;
                global_j = warp_begin_col + local_j;
                if (global_i < n_color && global_j < m_rhs) {
                    temp1 = loader_ptr1[IDX2D(global_i, global_j, n_color)];
                    temp2 = loader_ptr2[IDX2D(global_i, global_j, n_color)];
                } else {
                    temp1.x = temp1.y = temp2.x = temp2.y = 0;
                }
                smem_t_real[j] = temp1.x + temp2.y;
                smem_t_imag[j] = temp1.y - temp2.x;
            }
            __syncwarp();

            smem_t_real = smem_mat_T + 2 * WMMA_K * WMMA_N;
            smem_t_imag = smem_mat_T + 3 * WMMA_K * WMMA_N;
            // load T2, T2 = M2 - iM3 = (M2.real + M3.imag) + i(M2.imag - M3.real)
            for (int j = threadIdx.x; j < WMMA_K * WMMA_N; j += WARP_SIZE) {
                loader_ptr1 = reinterpret_cast<Float2*>(global_fermion_in) + 1 * n_color * m_rhs;
                loader_ptr2 = reinterpret_cast<Float2*>(global_fermion_in) + 2 * n_color * m_rhs;
                local_i = j / WMMA_N;
                local_j = j % WMMA_N;
                global_i = i * WMMA_K + local_i;
                global_j = warp_begin_col + local_j;
                if (global_i < n_color && global_j < m_rhs) {
                    temp1 = loader_ptr1[IDX2D(global_i, global_j, n_color)];
                    temp2 = loader_ptr2[IDX2D(global_i, global_j, n_color)];
                } else {
                    temp1.x = temp1.y = temp2.x = temp2.y = 0;
                }
                smem_t2_real[j] = temp1.x + temp2.y;
                smem_t2_imag[j] = temp1.y - temp2.x;
            }
            // sync
            __syncwarp();
            // TODO: 这个地方需要个register使用量的平衡，更多register,更少shared memory存取
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A1_frag;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A2_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B1_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B2_frag;
            // matmul
            // load U from smem to register
            wmma::load_matrix_sync(A1_frag, smem_u_real, WMMA_K);
            wmma::load_matrix_sync(A2_frag, smem_u_imag, WMMA_K);
            // U T1 = [U_real * T1_real - U_imag * T1_imag, U_real * T1_imag + U_imag * T1_real]
            smem_r_real = smem_mat_R;                    // R1_real
            smem_r_imag = smem_mat_R + WMMA_K * WMMA_N;  // R1_imag
            smem_t_real = smem_mat_T;                    // T1_real
            smem_t_imag = smem_mat_T + WMMA_M * WMMA_N;  // T1_imag
            // load fermion T1 from smem to register
            wmma::load_matrix_sync(C_frag, smem_r_real, WMMA_N);
            wmma::load_matrix_sync(B1_frag, smem_t_real, WMMA_N);
            wmma::load_matrix_sync(B2_frag, smem_t_imag, WMMA_N);
            // R1_real = U_real * T1_real - U_imag * T1_imag
            // right side
            wmma::mma_sync(C_frag, A2_frag, B2_frag, C_frag);                           // U_imag * T1_imag
            wmma::store_matrix_sync(smem_r_real, C_frag, WMMA_N, wmma::mem_row_major);  // store to shared memory
            for (int j = threadIdx.x; j < WMMA_M * WMMA_N; j += WARP_SIZE) {            // r_elem = - r_elem
                smem_r_real[j] = -smem_r_real[j];
            }
            wmma::load_matrix_sync(C_frag, smem_r_real, WMMA_N);  // C1 = -U_imag * T1_imag
            // left_side
            wmma::mma_sync(C_frag, A1_frag, B1_frag, C_frag);  // C1 <--- U_real * T1_real - U_imag * T1_imag
            wmma::store_matrix_sync(smem_r_real, C_frag, WMMA_N, wmma::mem_row_major);  // store to shared memory
            // R1_imag = U_real * T1_imag + U_imag * T1_real
            wmma::load_matrix_sync(C_frag, smem_r_imag, WMMA_N);
            wmma::mma_sync(C_frag, A2_frag, B1_frag, C_frag);
            wmma::mma_sync(C_frag, A2_frag, B1_frag, C_frag);  // C1 = U_imag * T1_real
            wmma::mma_sync(C_frag, A1_frag, B2_frag, C_frag);  // C1 = U_real * T1_imag + U_imag * T1_real
            wmma::store_matrix_sync(smem_r_imag, C_frag, WMMA_N, wmma::mem_row_major);  // store to shared memory

            // ----------------------separator----------------------
            // U T2 = [U_real * T2_real - U_imag * T2_imag, U_real * T2_imag + U_imag * T2_real]
            smem_r_real = smem_mat_R + 2 * WMMA_K * WMMA_N;  // R2_real
            smem_r_imag = smem_mat_R + 3 * WMMA_K * WMMA_N;  // R2_imag
            smem_t_real = smem_mat_T + 2 * WMMA_M * WMMA_N;  // T2_real
            smem_t_imag = smem_mat_T + 3 * WMMA_M * WMMA_N;  // T2_imag
            // load fermion T2 from smem to register
            wmma::load_matrix_sync(B1_frag, smem_t_real, WMMA_N);
            wmma::load_matrix_sync(B2_frag, smem_t_imag, WMMA_N);
            // R2_real = U_real * T2_real - U_imag * T2_imag
            // right side
            wmma::mma_sync(C_frag, A2_frag, B2_frag, C_frag);                           // U_imag * T2_imag
            wmma::store_matrix_sync(smem_r_real, C_frag, WMMA_N, wmma::mem_row_major);  // store to shared memory
            for (int j = threadIdx.x; j < WMMA_M * WMMA_N; j += WARP_SIZE) {            // r_elem = - r_elem
                smem_r_real[j] = -smem_r_real[j];
            }
            wmma::load_matrix_sync(C_frag, smem_r_real, WMMA_N);  // C1 = -U_imag * T2_imag
            // left_side
            wmma::mma_sync(C_frag, A1_frag, B1_frag, C_frag);  // C1 <--- U_real * T2_real - U_imag * T2_imag
            wmma::store_matrix_sync(smem_r_real, C_frag, WMMA_N, wmma::mem_row_major);  // store to shared memory
            // R2_imag = U_real * T2_imag + U_imag * T2_real
            wmma::load_matrix_sync(C_frag, smem_r_imag, WMMA_N);
            wmma::mma_sync(C_frag, A2_frag, B1_frag, C_frag);  // C1 = U_imag * T2_real
            wmma::mma_sync(C_frag, A1_frag, B2_frag, C_frag);  // C1 = U_real * T2_imag + U_imag * T2_real
            wmma::store_matrix_sync(smem_r_imag, C_frag, WMMA_N, wmma::mem_row_major);  // store to shared memory

        }  // for i : K_TILES
        // calc partial R1 R2 over
        // add R to T1
        // x, fwd:  L1 = L1 + R1, L2 = L2 + R2, L3 = L3 + iR2, L4 = L4 + iR1
        AccFloat* smem_l_real;
        AccFloat* smem_l_imag;

        // L1, L4
        smem_r_real = smem_mat_R;                    // R1_real
        smem_r_imag = smem_mat_R + WMMA_M * WMMA_N;  // R1_imag
        for (int j = 0; j < WMMA_M * WMMA_N; j += WARP_SIZE) {
            temp1.x = smem_r_real[j];  // R1.real
            temp1.y = smem_r_imag[j];  // R1.imag
            // L1
            smem_l_real = smem_mat_L + 0 * (2 * WMMA_M * WMMA_N);  // L1_real
            smem_l_imag = smem_l_real + WMMA_M * WMMA_N;           // L1_imag
            temp2.x = smem_l_real[j];
            temp2.y = smem_l_imag[j];
            smem_l_real[j] = temp2.x + temp1.x;  // L1.real = L1.real + R1.real
            smem_l_imag[j] = temp2.y + temp1.y;  // L1.imag = L1.imag + R1.imag

            // L4
            smem_l_real = smem_mat_L + 3 * (2 * WMMA_M * WMMA_N);  // L4_real
            smem_l_imag = smem_l_real + WMMA_M * WMMA_N;           // L4_imag
            temp2.x = smem_l_real[j];
            temp2.y = smem_l_imag[j];
            smem_l_real[j] = temp2.x - temp1.y;  // L4.real = L4.real - R1.imag
            smem_l_imag[j] = temp2.y + temp1.x;  // L4.imag = L4.imag + R1.real
        }
        // L2, L3, use R2
        smem_r_real = smem_mat_R + 2 * WMMA_N * WMMA_N;  // R2_real
        smem_r_imag = smem_mat_R + 3 * WMMA_M * WMMA_N;  // R2_imag
        for (int j = 0; j < WMMA_M * WMMA_N; j += WARP_SIZE) {
            temp1.x = smem_r_real[j];  // R2.real
            temp1.x = smem_r_imag[j];  // R2.imag
            // L2 = L2 + R2
            smem_l_real = smem_mat_L + 1 * (2 * WMMA_M * WMMA_N);  // L2.real
            smem_l_imag = smem_l_real + WMMA_M * WMMA_N;           // L2.imag
            temp2.x = smem_l_real[j];
            temp2.y = smem_l_imag[j];
            smem_l_real[j] = temp2.x + temp1.x;  // L2.real = L2.real + R2.real
            smem_l_imag[j] = temp2.y + temp1.y;  // L2.imag = L2.imag + R2.imag

            // L3 = L3 + iR2
            smem_l_real = smem_mat_L + 2 * (2 * WMMA_M * WMMA_N);  // L3_real
            smem_l_imag = smem_l_real + WMMA_M * WMMA_N;           // L3_imag
            temp2.x = smem_l_real[j];
            temp2.y = smem_l_imag[j];
            smem_l_real[j] = temp2.x - temp1.y;  // L2.real = L3.real - R2.imag
            smem_l_imag[j] = temp2.y + temp1.x;  // L2.imag = L3.imag + R2.real
        }
        // store this tile to global memory
    }
}

}  // namespace device
}  // namespace qcu