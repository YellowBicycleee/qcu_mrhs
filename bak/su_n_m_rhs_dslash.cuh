#pragma once
#include <cuda_fp16.h>

// #include "kernel/constants.cuh"
#include "kernel/su_n_m_rhs_matmul.cuh"
#include "point/qcu_point.cuh"
#include "qcu_float_float2_wrapper.h"
#include "qcu_utils.h"
#include "qcu_wmma_constant.h"

namespace qcu {
namespace device {

template <typename Float>
__device__ void single_point_wilson_dslash(Float* __restrict__ out, Float* __restrict__ in, Float* __restrict__ gauge,
                                           Float* smem, int Lx, int Ly, int Lz, int Lt, int g_x, int g_y, int g_z,
                                           int g_t, int parity, bool dagger_flag, int n_color, int m_rhs,
                                           int virtual_point_id) {
    using Float2 = typename qcu::Float2Wrapper<Float>::Float2;
    constexpr int WMMA_M = WMMA_Param<Float>::WMMA_M;
    constexpr int WMMA_N = WMMA_Param<Float>::WMMA_N;
    constexpr int WMMA_K = WMMA_Param<Float>::WMMA_K;

    int half_Lx = Lx / 2;

    int t = virtual_point_id / (Lz * Ly * half_Lx);
    int z = virtual_point_id % (Lz * Ly * half_Lx) / (Ly * half_Lx);
    int y = virtual_point_id % (Ly * half_Lx) / half_Lx;
    int x = virtual_point_id % half_Lx;

    int real_warp_cnt = blockDim.y;
    int warp_id = threadIdx.y;

    // clang-format off
    constexpr int smem_float_elems_per_warp = 2 * (
                                       WMMA_M * WMMA_K +         /* U                          wmma_m * wmma_k */
                                       WMMA_K * WMMA_N +         /* T1                         wmma_k * wmma_n */
                                 6 * (WMMA_M * WMMA_N)           /* R1, R2, L1, L2, L3, L4     wmma_m * wmma_n */
                                );
    
    Float* warp_smem = smem + warp_id * smem_float_elems_per_warp;
    Float* smem_U    = warp_smem;                          // U
    Float* smem_T    = smem_U    + WMMA_M * WMMA_K * 2;       // T     -------  T1, T2 share the same memory
    Float* smem_R    = smem_T    + WMMA_K * WMMA_N * 2;       // R1, R2
    Float* smem_L    = smem_R    + WMMA_M * WMMA_N * 2 * 2;   // L1, L2, L3, L4
    // clang-format on

    int total_warps_row = div_ceil(n_color, WMMA_M);  // n color separate to M
    int total_warps_col = div_ceil(m_rhs, WMMA_N);    // m rhs separate to N
    int total_warps = total_warps_row * total_warps_col;

    Point point(x, y, z, t, parity);  // sink of point to calc
    Point mv_point;

    Float* point_gauge_matrix;  // gauge of target point
    Float* point_in_matrix;     // fermion in of target point
    Float* point_out_matrix;    // fermion out of target point

    for (int virtual_warp_id = warp_id; virtual_warp_id < total_warps; virtual_warp_id += real_warp_cnt) {
        int virtual_warp_id_i = virtual_warp_id / total_warps_col;
        int virtual_warp_id_j = virtual_warp_id % total_warps_col;

        int warp_begin_row = virtual_warp_id_i * WMMA_M;
        int warp_begin_col = virtual_warp_id_j * WMMA_N;

        // clear L[1, 2, 3, 4] (real and imag part)
        for (int i = threadIdx.x; i < WMMA_M * WMMA_N; i += WARP_SIZE) {
            int local_i = i / WMMA_N;
            // int local_j = i % WMMA_N;
            int local_j = i & (WMMA_N - 1);
            // if (blockIdx.x ==0) {
            //     printf("i = %d, j = %d\n", local_i, local_j);
            // }
            smem_L[IDX3D(0, local_i, local_j, WMMA_M, WMMA_N)] = 0;  // L1 real
            smem_L[IDX3D(1, local_i, local_j, WMMA_M, WMMA_N)] = 0;  // L1 imag
            smem_L[IDX3D(2, local_i, local_j, WMMA_M, WMMA_N)] = 0;  // L2 real
            smem_L[IDX3D(3, local_i, local_j, WMMA_M, WMMA_N)] = 0;  // L2 imag
            smem_L[IDX3D(4, local_i, local_j, WMMA_M, WMMA_N)] = 0;  // L3 real
            smem_L[IDX3D(5, local_i, local_j, WMMA_M, WMMA_N)] = 0;  // L3 imag
            smem_L[IDX3D(6, local_i, local_j, WMMA_M, WMMA_N)] = 0;  // L4 real
            smem_L[IDX3D(7, local_i, local_j, WMMA_M, WMMA_N)] = 0;  // L4 imag
        }
        __syncwarp();
        // // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> L1_real_frag;
        // // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> L1_imag_frag;
        // // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> L2_real_frag;
        // // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> L2_imag_frag;
        // // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> L3_real_frag;
        // // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> L3_imag_frag;
        // // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> L4_real_frag;
        // // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float> L4_imag_frag;
        // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Float>[8] L_frag;
        // for (int i = 0; i < 8; i++) {}

        // x fwd
        mv_point = point.move(FWD, X_DIM, half_Lx, Ly, Lz, Lt);
        point_gauge_matrix = point.getGaugeAddr(gauge, X_DIM, half_Lx, Lt, Lz, Lt, n_color);
        point_in_matrix = mv_point.getGatheredColorSpinorAddr(in, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        dslash_mat_mul<Float>(smem_L, smem_U, smem_R, smem_T, point_gauge_matrix, point_in_matrix, dagger_flag, n_color,
                              m_rhs, warp_begin_row, warp_begin_col, X_DIM, FWD);
        // x bwd
        mv_point = point.move(BWD, X_DIM, half_Lx, Ly, Lz, Lt);
        point_gauge_matrix = mv_point.getGaugeAddr(gauge, X_DIM, half_Lx, Lt, Lz, Lt, n_color);
        point_in_matrix = mv_point.getGatheredColorSpinorAddr(in, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        dslash_mat_mul<Float>(smem_L, smem_U, smem_R, smem_T, point_gauge_matrix, point_in_matrix, dagger_flag, n_color,
                              m_rhs, warp_begin_row, warp_begin_col, X_DIM, BWD);

        // y fwd
        mv_point = point.move(FWD, Y_DIM, half_Lx, Ly, Lz, Lt);
        point_gauge_matrix = point.getGaugeAddr(gauge, Y_DIM, half_Lx, Lt, Lz, Lt, n_color);
        point_in_matrix = mv_point.getGatheredColorSpinorAddr(in, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        dslash_mat_mul<Float>(smem_L, smem_U, smem_R, smem_T, point_gauge_matrix, point_in_matrix, dagger_flag, n_color,
                              m_rhs, warp_begin_row, warp_begin_col, Y_DIM, FWD);
        // y bwd
        mv_point = point.move(BWD, Y_DIM, half_Lx, Ly, Lz, Lt);
        point_gauge_matrix = mv_point.getGaugeAddr(gauge, Y_DIM, half_Lx, Lt, Lz, Lt, n_color);
        point_in_matrix = mv_point.getGatheredColorSpinorAddr(in, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        dslash_mat_mul<Float>(smem_L, smem_U, smem_R, smem_T, point_gauge_matrix, point_in_matrix, dagger_flag, n_color,
                              m_rhs, warp_begin_row, warp_begin_col, Y_DIM, BWD);

        // z fwd
        mv_point = point.move(FWD, Z_DIM, half_Lx, Ly, Lz, Lt);
        point_gauge_matrix = point.getGaugeAddr(gauge, Z_DIM, half_Lx, Lt, Lz, Lt, n_color);
        point_in_matrix = mv_point.getGatheredColorSpinorAddr(in, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        dslash_mat_mul<Float>(smem_L, smem_U, smem_R, smem_T, point_gauge_matrix, point_in_matrix, dagger_flag, n_color,
                              m_rhs, warp_begin_row, warp_begin_col, Z_DIM, FWD);
        // z bwd
        mv_point = point.move(BWD, Z_DIM, half_Lx, Ly, Lz, Lt);
        point_gauge_matrix = mv_point.getGaugeAddr(gauge, Z_DIM, half_Lx, Lt, Lz, Lt, n_color);
        point_in_matrix = mv_point.getGatheredColorSpinorAddr(in, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        dslash_mat_mul<Float>(smem_L, smem_U, smem_R, smem_T, point_gauge_matrix, point_in_matrix, dagger_flag, n_color,
                              m_rhs, warp_begin_row, warp_begin_col, Z_DIM, BWD);

        // t fwd
        mv_point = point.move(FWD, T_DIM, half_Lx, Ly, Lz, Lt);
        point_gauge_matrix = point.getGaugeAddr(gauge, T_DIM, half_Lx, Lt, Lz, Lt, n_color);
        point_in_matrix = mv_point.getGatheredColorSpinorAddr(in, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        dslash_mat_mul<Float>(smem_L, smem_U, smem_R, smem_T, point_gauge_matrix, point_in_matrix, dagger_flag, n_color,
                              m_rhs, warp_begin_row, warp_begin_col, T_DIM, FWD);
        // t bwd
        mv_point = point.move(BWD, T_DIM, half_Lx, Ly, Lz, Lt);
        point_gauge_matrix = mv_point.getGaugeAddr(gauge, T_DIM, half_Lx, Lt, Lz, Lt, n_color);
        point_in_matrix = mv_point.getGatheredColorSpinorAddr(in, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        dslash_mat_mul<Float>(smem_L, smem_U, smem_R, smem_T, point_gauge_matrix, point_in_matrix, dagger_flag, n_color,
                              m_rhs, warp_begin_row, warp_begin_col, T_DIM, BWD);

        // store L back to fermion out (global memory)
        point_out_matrix = point.getGatheredColorSpinorAddr(out, half_Lx, Lt, Lz, Lt, n_color, m_rhs);
        warp_store_complex_from_smem_to_global(point_out_matrix, warp_begin_row, warp_begin_col, n_color, m_rhs,
                                               smem_L + 0 * WMMA_M * WMMA_N * 2, WMMA_M, WMMA_N);  // L1
        warp_store_complex_from_smem_to_global(point_out_matrix + 1 * 2 * n_color * m_rhs, warp_begin_row,
                                               warp_begin_col, n_color, m_rhs, smem_L + 1 * WMMA_M * WMMA_N * 2, WMMA_M,
                                               WMMA_N);  // L2
        warp_store_complex_from_smem_to_global(point_out_matrix + 2 * 2 * n_color * m_rhs, warp_begin_row,
                                               warp_begin_col, n_color, m_rhs, smem_L + 2 * WMMA_M * WMMA_N * 2, WMMA_M,
                                               WMMA_N);  // L3
        warp_store_complex_from_smem_to_global(point_out_matrix + 3 * 2 * n_color * m_rhs, warp_begin_row,
                                               warp_begin_col, n_color, m_rhs, smem_L + 3 * WMMA_M * WMMA_N * 2, WMMA_M,
                                               WMMA_N);  // L4
        __syncwarp();

    }  // end for
}

template <typename Float>
__global__ void wilson_dslash_su_n_mrhs(Float* __restrict__ out, Float* __restrict__ in, Float* __restrict__ gauge,
                                        int Lx, int Ly, int Lz, int Lt, int g_x, int g_y, int g_z, int g_t, int parity,
                                        bool dagger_flag, int n_color, int m_rhs) {
    // block切分使用2D，dim3(WARP_SIZE, WARP_NUMBER)
    int block_id = blockIdx.x;
    int grid_size = gridDim.x;  // 1D grid
    int vol = Lx * Ly * Lz * Lt / 2;

    // 这里假设每个warp存储一堆数据，一个warp内部的数据排列是
    //    U  --------- WMMA_M * WMMA_K * 2    (WMMA_M * WMMA_K real + WMMA_M * WMMA_K imag)
    //    T  --------- WMMA_K * WMMA_N * 2    (WMMA_K * WMMA_N real + WMMA_K * WMMA_N imag)
    //    R1 --------- WMMA_M * WMMA_N * 2    (WMMA_M * WMMA_N real + WMMA_M * WMMA_N imag)
    //    R2 --------- WMMA_M * WMMA_N * 2    (WMMA_M * WMMA_N real + WMMA_M * WMMA_N imag)
    //    L1 --------- WMMA_M * WMMA_N * 2    (WMMA_M * WMMA_N real + WMMA_M * WMMA_N imag)
    //    L2 --------- WMMA_M * WMMA_N * 2    (WMMA_M * WMMA_N real + WMMA_M * WMMA_N imag)
    //    L3 --------- WMMA_M * WMMA_N * 2    (WMMA_M * WMMA_N real + WMMA_M * WMMA_N imag)
    //    L4 --------- WMMA_M * WMMA_N * 2    (WMMA_M * WMMA_N real + WMMA_M * WMMA_N imag)
    // 每个warp需要的共享内存数量是
    // 2 * (WMMAM * WMMA_K + WMMA_K * WMMA_N) + 2 * 6 * WMMA_M * WMMA_N

    // clang-format off
    constexpr int WMMA_M = WMMA_Param<Float>::WMMA_M;
    constexpr int WMMA_N = WMMA_Param<Float>::WMMA_N;
    constexpr int WMMA_K = WMMA_Param<Float>::WMMA_K;
    constexpr int smem_float_elems_per_warp = 2 * (
                                       WMMA_M * WMMA_K +         /* U                          wmma_m * wmma_k */
                                       WMMA_K * WMMA_N +         /* T1                         wmma_k * wmma_n */
                                 6 * (WMMA_M * WMMA_N)           /* R1, R2, L1, L2, L3, L4     wmma_m * wmma_n */
                                );
    __shared__ Float smem[WARP_PER_BLOCK * smem_float_elems_per_warp];  // smem数量从外部统一指定

    // clang-format off
    for (int i = block_id; i < vol; i += grid_size) {
        single_point_wilson_dslash<Float>(  out, in, gauge, smem, Lx, Ly, Lz, Lt, g_x, g_y, g_z, g_t, 
                                            parity, dagger_flag, n_color, m_rhs, i);
    }
    // clang-format on
}

}  // namespace device
}  // namespace qcu