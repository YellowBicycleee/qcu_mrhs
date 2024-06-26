#pragma once
#include <mma.h>

#include <cassert>

#include "qcu_enum.h"
#include "qcu_float_float2_wrapper.h"
#include "qcu_macro.h"
#include "qcu_wmma_constant.h"
namespace qcu {
namespace device {
using namespace nvcuda;
// clang-format off

// 只计算(1-\gamma), 此时dagger_flag=0，（1+\gamma）时dagger_flag=1即可求得
// single warp, for a single 16 * 16 fp16 or 8 * 4 fp64 matrix
template <typename Float>
__device__ __forceinline__ void load_complex_fermion_mat_T1_from_global_to_smem (  // row-major
    Float* __restrict__ smem_T, int smem_k, int smem_n,
    const Float* __restrict__ global_mem, int global_iter_start_k, int global_iter_start_n, /* complex/float2 storage */
    int gamma_idx /* 1, 2, 3, 4 */ , int dagger_flag /* 0 1 */, int n_color, int m_rhs) 
{
    // smem_T 布局：
    // Float: smem_T1_real  0                  -----(    smem_k * smem_n - 1)
    // Float: smem_T1_imag      smem_k * smem_n-----(2 * smem_k * smem_n - 1)
    // Float: smem_T2_real  2 * smem_k * smem_n-----(3 * smem_k * smem_n - 1)
    // Float: smem_T2_imag  3 * smem_k * smem_n-----(4 * smem_k * smem_n - 1)

    using Float2 = typename qcu::Float2Wrapper<Float>::Float2;    

    for (int idx = threadIdx.x; idx < smem_k * smem_n; idx += WARP_SIZE) {
        // smem is always row-major
        int smem_i = idx / smem_n;
        int smem_j = idx % smem_n;
        int global_i = global_iter_start_k + smem_i;
        int global_j = global_iter_start_n + smem_j;
        Float2 temp1;
        Float2 temp2;

        if (global_i >= n_color || global_j >= m_rhs) {   // 0 - padding
            smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = Float(0.0);  // T1.real
            smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = Float(0.0);  // T1.imag
            // smem_T[IDX3D(2, smem_i, smem_j, smem_k, smem_n)] = Float(0.0);  // T2.real
            // smem_T[IDX3D(3, smem_i, smem_j, smem_k, smem_n)] = Float(0.0);  // T2.imag
        }
        else {
            switch (gamma_idx) {
            case 1:
                {
                    // T1 = M1 - iM4
                    temp1 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(0, global_i, global_j, n_color, m_rhs)]; // M1 elem
                    temp2 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(3, global_i, global_j, n_color, m_rhs)]; // M4 elem
                    // combine and store to smem
                    if (dagger_flag == 0) {                                         // T1 = M1 - iM4
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x + temp2.y;  // M1.real + M4.imag
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y - temp2.x;  // M1.imag - M4.real
                    } else {                                                        // T1 = M1 + iM4
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x - temp2.y;  // M1.real - M4.imag
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp2.y + temp1.y;  // M1.imag + M4.imag
                    }
                }

                break;

            case 2: 
                {   // T1 = M1 + M4
                    temp1 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(0, global_i, global_j, n_color, m_rhs)]; // M1 elem
                    temp2 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(3, global_i, global_j, n_color, m_rhs)]; // M4 elem
                    
                    // combine and store to smem
                    if (dagger_flag == 0) {
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x + temp2.x;  // M1.real + M4.real
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y + temp2.y;  // M1.imag + M4.imag
                    } else {
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x - temp2.x;  // M1.real - M4.real
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y - temp2.y;  // M1.imag - M4.imag
                    }

                } 
                break;

            case 3:
                {   // T1 = M1 - iM3
                    temp1 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(0, global_i, global_j, n_color, m_rhs)]; // M1 elem
                    temp2 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(2, global_i, global_j, n_color, m_rhs)]; // M3 elem

                    // combine and store to smem
                    if (dagger_flag == 0) {
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x + temp2.y;  // T1.real = M1.real + M3.imag
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y - temp2.x;  // T1.imag = M1.imag - M3.real
                    } else {
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x - temp2.y;  // T1.real = M1.real - M3.imag
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y + temp2.x;  // T1.imag = M1.imag + M3.real
                    }
                }
                break;

            case 4:
                {   // T1 = M1 - M3, T2 = M2 - M4

                    // T1 = M1 - M3
                    temp1 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(0, global_i, global_j, n_color, m_rhs)]; // M1 elem
                    temp2 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(2, global_i, global_j, n_color, m_rhs)]; // M3 elem
                    // combine and store to smem
                    if (dagger_flag == 0) {  // T1 = M1 - M3
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x - temp2.x;  // T1.real = M1.real - M3.real
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y - temp2.y;  // T1.imag = M1.imag - M3.imag
                    } else {            // T1 = M1 + M3
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x + temp2.x;  // T1.real = M1.real + M3.real
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y + temp2.y;  // T1.imag = M1.imag + M3.imag
                    }
                }
                break;

            default:
                assert(0);
                break;
            } // end switch
        } // end if-else 
    } // end for
    __syncwarp();
}
// clang-format on

// clang-format off

// 只计算(1-\gamma), 此时dagger_flag=0，（1+\gamma）时dagger_flag=1即可求得
// single warp, for a single 16 * 16 fp16 or 8 * 4 fp64 matrix
template <typename Float>
__device__ __forceinline__ void load_complex_fermion_mat_T2_from_global_to_smem(  // row-major
    Float* __restrict__ smem_T, int smem_k, int smem_n,
    const Float* __restrict__ global_mem, int global_iter_start_r, int global_iter_start_n, /* complex/float2 storage */
    int gamma_idx /* 1, 2, 3, 4 */ , int dagger_flag /* 0 1 */, int n_color, int m_rhs) 
{
    // smem_T 布局：
    // Float: smem_T_real  0                  -----(    smem_k * smem_n - 1)
    // Float: smem_T_imag      smem_k * smem_n-----(2 * smem_k * smem_n - 1)
    using Float2 = typename qcu::Float2Wrapper<Float>::Float2;
    
    for (int idx = threadIdx.x; idx < smem_k * smem_n; idx += WARP_SIZE) {
        // smem is always row-major
        int smem_i = idx / smem_n;
        int smem_j = idx % smem_n;
        int global_i = global_iter_start_r + smem_i;
        int global_j = global_iter_start_n + smem_j;
        Float2 temp1;
        Float2 temp2;

        if (global_i >= n_color || global_j >= m_rhs) {   // 0 - padding
            smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = Float(0.0);  // T.real
            smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = Float(0.0);  // T.imag
        }
        else {
            switch (gamma_idx) {
            case 1:
                {
                    // T2 = M2 - iM3
                    temp1 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(1, global_i, global_j, n_color, m_rhs)]; // M2 elem
                    temp2 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(2, global_i, global_j, n_color, m_rhs)]; // M3 elem
                   
                    // combine and store to smem
                    if (dagger_flag == 0) {
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x + temp2.y;  // M2.real + M3.imag
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y - temp2.x;  // M2.imag - M3.real 
                    } else {
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x - temp2.y;  // M2.real - M3.imag
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp2.y + temp1.y;  // M2.imag + M3.imag
                    }
                }

                break;

            case 2: 
                {   // T2 = M2 - M3
                    temp1 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(1, global_i, global_j, n_color, m_rhs)]; // M2 elem
                    temp2 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(2, global_i, global_j, n_color, m_rhs)]; // M3 elem
                    // combine and store to smem
                    if (dagger_flag == 0) {
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x - temp2.x;  // M2.real - M3.real
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y - temp2.y;  // M2.imag - M3.imag
                    } else {
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x + temp1.x;  // M2.real + M3.real
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp2.x + temp1.y;  // M2.imag + M3.imag
                    }

                } 
                break;

            case 3:
                {   // T2 = M2 + iM4
                    // read or padding
                    temp1 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(1, global_i, global_j, n_color, m_rhs)]; // M2 elem
                    temp2 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(3, global_i, global_j, n_color, m_rhs)]; // M4 elem
                    
                    // combine and store to smem
                    if (dagger_flag == 0) {  // T2 = M2 + iM4
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x + temp2.y;  // T2.real = M2.real - M4.imag
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y + temp2.x;  // T2.imag = M2.imag + M4.real
                    } else {            // T2 = M2 - iM4
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x - temp2.y;  // T2.real = M2.real + M4.imag
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y - temp2.x;  // T2.imag = M2.imag - M4.real
                    }
                }
                break;

            case 4:
                {   // T2 = M2 - M4
                    temp1 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(1, global_i, global_j, n_color, m_rhs)]; // M2 elem
                    temp2 = reinterpret_cast<const Float2*>(global_mem)[IDX3D(3, global_i, global_j, n_color, m_rhs)]; // M4 elem

                    // combine and store to smem
                    if (dagger_flag == 0) {  // T2 = M2 - M4
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x - temp2.x;  // T2.real = M2.real - M4.real
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y - temp2.y;  // T2.imag = M2.imag - M4.imag
                    } else {            // T2 = M2 + M4
                        smem_T[IDX3D(0, smem_i, smem_j, smem_k, smem_n)] = temp1.x + temp2.x;  // T2.real = M2.real + M4.real
                        smem_T[IDX3D(1, smem_i, smem_j, smem_k, smem_n)] = temp1.y + temp2.y;  // T2.imag = M2.imag + M4.imag
                    }

                }
                break;

            default:
                assert(0);
                break;
            } // end switch
        } // end if-else 
    } // end for
    __syncwarp();
}
// clang-format on

// clang-format off
// global_mem: row_major
// single warp!!!!
template <typename Float> 
__device__ __forceinline__ void load_complex_gauge_mat_from_global_to_smem ( 
    Float* __restrict__ smem_U, int smem_m, int smem_k,
    const Float* __restrict__ global_mem, int global_iter_start_m, int global_iter_start_n, /* complex/float2 storage */
    bool if_dagger, int n_color  
) {
    using Float2 = typename qcu::Float2Wrapper<Float>::Float2;

    for (int idx = threadIdx.x; idx < smem_m * smem_k; idx += WARP_SIZE) {
        // smem is always row-major
        int smem_i = idx / smem_k;
        int smem_j = idx % smem_k;
        int global_i;
        int global_j;
        Float2 temp;
    
        if (!if_dagger) {
            global_i = global_iter_start_m + smem_i;
            global_j = global_iter_start_n + smem_j;
            if (global_i < n_color && global_j < n_color) {
                temp = reinterpret_cast<const Float2*>(global_mem)[IDX3D(0, global_i, global_j, n_color, n_color)]; // U elem
            } else {
                temp.x = temp.y = 0.0;  // padding
            }
            smem_U[IDX3D(0, smem_i, smem_j, smem_m, smem_k)] = temp.x;
            smem_U[IDX3D(1, smem_i, smem_j, smem_m, smem_k)] = temp.y;
        } else {    // dagger
            global_i = global_iter_start_n + smem_j;
            global_j = global_iter_start_m + smem_i;

            if (global_i < n_color && global_j < n_color) {
                temp = reinterpret_cast<const Float2*>(global_mem)[IDX3D(0, global_i, global_j, n_color, n_color)]; // U elem
            } else {
                temp.x = temp.y = 0.0;  // padding
            }
            smem_U[IDX3D(0, smem_i, smem_j, smem_m, smem_k)] = temp.x;
            smem_U[IDX3D(1, smem_i, smem_j, smem_m, smem_k)] = -temp.y;
        }
        
    }
    __syncwarp();
}
// clang-format on

template <typename Float>
__device__ __forceinline__ void calc_L_from_R1(Float* __restrict__ smem_L, const Float* __restrict__ smem_R, int smem_m,
                                               int smem_n, int gamma_idx, bool dagger_flag) {
    // smem_L 布局：
    // L1.real                    0-----(    smem_m * smem_n - 1)
    // L1.imag      smem_m * smem_n-----(2 * smem_m * smem_n - 1)
    // L2.real  2 * smem_m * smem_n-----(3 * smem_m * smem_n - 1)
    // L2.imag  3 * smem_m * smem_n-----(4 * smem_m * smem_n - 1)
    // L3.real  4 * smem_m * smem_n-----(5 * smem_m * smem_n - 1)
    // L3.imag  5 * smem_m * smem_n-----(6 * smem_m * smem_n - 1)
    // L4.real  6 * smem_m * smem_n-----(7 * smem_m * smem_n - 1)
    // L4.imag  7 * smem_m * smem_n-----(8 * smem_m * smem_n - 1)

    using Float2 = typename qcu::Float2Wrapper<Float>::Float2;

    for (int idx = threadIdx.x; idx < smem_m * smem_n; idx += WARP_SIZE) {
        int smem_i = idx / smem_n;
        int smem_j = idx % smem_n;

        Float2 temp;
        // depend on R1
        temp.x = smem_R[IDX3D(0, smem_i, smem_j, smem_m, smem_n)];  //  R1.real
        temp.y = smem_R[IDX3D(1, smem_i, smem_j, smem_m, smem_n)];  //  R1.imag
        __syncwarp();
        // L1 = L1 + R1
        smem_L[IDX3D(0, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L1.real += R1.real
        smem_L[IDX3D(1, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L1.imag += R1.imag
        switch (gamma_idx) {
            case 1: {                                                            // L4 = L4 + dagger_flag_sgn * iR1
                if (!dagger_flag) {                                              // L4 = L4 + iR1
                    smem_L[IDX3D(6, smem_i, smem_j, smem_m, smem_n)] -= temp.y;  // L4.real -= R1.imag
                    smem_L[IDX3D(7, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L4.imag += R1.real
                } else {                                                         // L4 = L4 - iR1
                    smem_L[IDX3D(6, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L4.real += R1.imag
                    smem_L[IDX3D(7, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L4.imag -= R1.real
                }
            } break;
            case 2: {                                                            // L4 = L4 + dagger_flag_sgn * R1
                if (!dagger_flag) {                                              // L4 = L4 + R1
                    smem_L[IDX3D(6, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L4.real += R1.real
                    smem_L[IDX3D(7, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L4.imag += R1.imag
                } else {                                                         // L4 = L4 - R1
                    smem_L[IDX3D(6, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L4.real -= R1.real
                    smem_L[IDX3D(7, smem_i, smem_j, smem_m, smem_n)] -= temp.y;  // L4.imag -= R1.imag
                }
            } break;
            case 3: {                                                            // L3 = L3 + iR1 * dagger_flag_sgn
                if (!dagger_flag) {                                              // L3 = L3 + iR1
                    smem_L[IDX3D(4, smem_i, smem_j, smem_m, smem_n)] -= temp.y;  // L3.real -= R1.imag
                    smem_L[IDX3D(5, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L3.imag += R1.real
                } else {                                                         // L3 = L3 - iR1
                    smem_L[IDX3D(4, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L3.real += R1.imag
                    smem_L[IDX3D(5, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L3.imag -= R1.real
                }

            } break;
            case 4: {                                                            // L3 = L3 - R1 * dagger_flag_sgn
                if (!dagger_flag) {                                              // L3 = L3 - R1
                    smem_L[IDX3D(4, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L3.real -= R1.real
                    smem_L[IDX3D(5, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L3.imag -= R1.imag
                } else {                                                         // L3 = L3 + R1
                    smem_L[IDX3D(4, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L3.real += R1.real
                    smem_L[IDX3D(5, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L3.imag += R1.imag
                }
            } break;
            default:
                assert(0);
                break;
        }  // end switch
        __syncwarp();
    }  // end for
}

template <typename Float>
__device__ __forceinline__ void calc_L_from_R2(Float* __restrict__ smem_L, const Float* __restrict__ smem_R, int smem_m,
                                               int smem_n, int gamma_idx, bool dagger_flag) {
    // smem_L 布局：
    // L1.real                    0-----(    smem_m * smem_n - 1)
    // L1.imag      smem_m * smem_n-----(2 * smem_m * smem_n - 1)
    // L2.real  2 * smem_m * smem_n-----(3 * smem_m * smem_n - 1)
    // L2.imag  3 * smem_m * smem_n-----(4 * smem_m * smem_n - 1)
    // L3.real  4 * smem_m * smem_n-----(5 * smem_m * smem_n - 1)
    // L3.imag  5 * smem_m * smem_n-----(6 * smem_m * smem_n - 1)
    // L4.real  6 * smem_m * smem_n-----(7 * smem_m * smem_n - 1)
    // L4.imag  7 * smem_m * smem_n-----(8 * smem_m * smem_n - 1)

    using Float2 = typename qcu::Float2Wrapper<Float>::Float2;

    for (int idx = threadIdx.x; idx < smem_m * smem_n; idx += WARP_SIZE) {
        int smem_i = idx / smem_n;
        int smem_j = idx % smem_n;

        Float2 temp;
        // load R2 elem
        temp.x = smem_R[IDX3D(2, smem_i, smem_j, smem_m, smem_n)];  //  R2.real
        temp.y = smem_R[IDX3D(3, smem_i, smem_j, smem_m, smem_n)];  //  R2.imag
        __syncwarp();
        // L2 = L2 + R2
        smem_L[IDX3D(2, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L2.real += R2.real
        smem_L[IDX3D(3, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L2.imag += R2.imag
        switch (gamma_idx) {
            case 1: {                                                            // L3 = L3 + iR2 * dagger_flag_sgn
                if (!dagger_flag) {                                              // L3 = L3 + iR2
                    smem_L[IDX3D(4, smem_i, smem_j, smem_m, smem_n)] -= temp.y;  // L3.real -= R2.imag
                    smem_L[IDX3D(5, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L3.imag += R2.real
                } else {                                                         // L3 = L3 - iR2
                    smem_L[IDX3D(4, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L3.real += R2.imag
                    smem_L[IDX3D(5, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L3.imag -= R2.real
                }
            } break;
            case 2: {                                                            // L3 = L3 - R2 * dagger_flag_sgn
                if (!dagger_flag) {                                              // L3 = L3 - R2
                    smem_L[IDX3D(4, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L3.real -= R2.real
                    smem_L[IDX3D(5, smem_i, smem_j, smem_m, smem_n)] -= temp.y;  // L3.imag -= R2.imag
                } else {                                                         // L3 = L3 + R2
                    smem_L[IDX3D(4, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L3.real += R2.real
                    smem_L[IDX3D(5, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L3.imag += R2.imag
                }
            } break;
            case 3: {                                                            // L4 = L4 - iR2 * dagger_flag_sgn
                if (!dagger_flag) {                                              // L4 = L4 - iR2
                    smem_L[IDX3D(6, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L4.real += R2.imag
                    smem_L[IDX3D(7, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L4.imag -= R2.real
                } else {
                    smem_L[IDX3D(6, smem_i, smem_j, smem_m, smem_n)] -= temp.y;  // L4.real -= R2.imag
                    smem_L[IDX3D(7, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L4.imag += R2.real
                }
            } break;
            case 4: {                                                            // L4 = L4 - R2 * dagger_flag_sgn
                if (!dagger_flag) {                                              // L4 = L4 - R2
                    smem_L[IDX3D(6, smem_i, smem_j, smem_m, smem_n)] -= temp.x;  // L4.real -= R2.real
                    smem_L[IDX3D(7, smem_i, smem_j, smem_m, smem_n)] -= temp.y;  // L4.imag -= R2.imag
                } else {                                                         // L4 = L4 + R2
                    smem_L[IDX3D(6, smem_i, smem_j, smem_m, smem_n)] += temp.x;  // L4.real += R2.real
                    smem_L[IDX3D(7, smem_i, smem_j, smem_m, smem_n)] += temp.y;  // L4.imag += R2.imag
                }
            } break;
            default:
                assert(0);
                break;
        }
    }
    __syncwarp();
}
// clang-format on
// smem_c : accumulator and output
// [Cr, Ci] += [Ar * Br - Ai * Bi, Ar * Bi + Ai * Br]
// Cr = Cr + Ar * Br - Ai * Bi
// Ci = Ci + Ar * Bi + Ai * Br
// C : M * N       A : M * K       B : K * N
template <typename Float>
__device__ __forceinline__ void tensor_core_complex_matmul(
    Float* A_shared, Float* B_shared,
    wmma::fragment<wmma::accumulator, WMMA_Param<Float>::WMMA_M, WMMA_Param<Float>::WMMA_N, WMMA_Param<Float>::WMMA_K,
                   Float>& C_real_frag,
    wmma::fragment<wmma::accumulator, WMMA_Param<Float>::WMMA_M, WMMA_Param<Float>::WMMA_N, WMMA_Param<Float>::WMMA_K,
                   Float>& C_imag_frag) {
    constexpr int WMMA_M = WMMA_Param<Float>::WMMA_M;
    constexpr int WMMA_N = WMMA_Param<Float>::WMMA_N;
    constexpr int WMMA_K = WMMA_Param<Float>::WMMA_K;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, Float, wmma::row_major> A_real_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, Float, wmma::row_major> A_imag_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, Float, wmma::row_major> B_real_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, Float, wmma::row_major> B_imag_frag;

    // load A, B
    wmma::load_matrix_sync(A_real_frag, A_shared, WMMA_K);
    wmma::load_matrix_sync(A_imag_frag, A_shared + WMMA_M * WMMA_K, WMMA_K);
    wmma::load_matrix_sync(B_real_frag, B_shared, WMMA_N);
    wmma::load_matrix_sync(B_imag_frag, B_shared + WMMA_K * WMMA_N, WMMA_N);

    // C imag
    wmma::mma_sync(C_imag_frag, A_real_frag, B_imag_frag, C_imag_frag);
    wmma::mma_sync(C_imag_frag, A_imag_frag, B_real_frag, C_imag_frag);

    // C real
    // Cr = Cr + (Ar * Br - Ai * Bi)
    // for every element in Ai, Ai = -Ai
    for (int i = 0; i < A_imag_frag.num_elements; i++) {
        A_imag_frag.x[i] = -A_imag_frag.x[i];
    }
    __syncwarp();
    wmma::mma_sync(C_real_frag, A_real_frag, B_real_frag, C_real_frag);  // Cr = Cr + Ar * Br
    wmma::mma_sync(C_real_frag, A_imag_frag, B_imag_frag, C_real_frag);  // Cr = Cr - Ai * Bi
}

template <typename Float>
__device__ __forceinline__ void warp_store_complex_from_smem_to_global(Float* __restrict__ global_mem,
                                                                       int global_tile_start_m, int global_tile_start_n,
                                                                       int global_total_m, int global_total_n,
                                                                       Float* warp_smem, int smem_m, int smem_n

) {
    using Float2 = typename qcu::Float2Wrapper<Float>::Float2;
    for (int i = threadIdx.x; i < smem_m * smem_n; i += WARP_SIZE) {
        int local_i = threadIdx.x / smem_n;
        int local_j = threadIdx.x % smem_n;
        int global_i = global_tile_start_m + local_i;
        int global_j = global_tile_start_n + local_j;

        if (global_i < global_total_m && global_j < global_total_n) {
            Float2 temp;
            temp.x = warp_smem[IDX3D(0, local_i, local_j, smem_m, smem_n)];
            temp.y = warp_smem[IDX3D(1, local_i, local_j, smem_m, smem_n)];
            reinterpret_cast<Float2*>(global_mem)[IDX2D(global_i, global_j, global_total_n)] = temp;
        }
    }
    __syncwarp();
}

}  // namespace device
}  // namespace qcu