#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "base/datatype/qcu_complex.cuh"
#include "base/datatype/qcu_float2.cuh"
#include "kernel/su_n_m_rhs_matmul.cuh"
#include "point/qcu_point.cuh"
#include "qcu_utils.h"
#include "qcu_wmma_constant.h"

namespace qcu {
namespace device {

constexpr int THREADS_PER_WARP_LINE = 8;
constexpr int WARP_LINES = 4;
// calculate 1 + gamma, if dagger, just set col(1) = -col(1)
// for example,
// ---------------------------------------
// 1 + gamma_1 = 
//      [ 1  0  0  i]
//      [ 0  1  i  0]
//      [ 0 -i  1  0]
//      [-i  0  0  1]   
// ---------------------------------------
// 1 + gamma_2 = 
//      [ 1  0  0 -1]
//      [ 0  1  1  0]
//      [ 0  1  1  0]
//      [-1  0  0  1]
// ---------------------------------------
// 1 + gamma_3 = 
//      [ 1  0  i   0]
//      [ 0  1  0  -i]
//      [-i  0  1   0]
//      [ 0  i  0   1]
// ---------------------------------------
// 1 + gamma_4 = 
//      [ 1  0  1  0]
//      [ 0  1  0  1]
//      [ 1  0  1  0]
//      [ 0  1  0  1]
// ---------------------------------------
// we can see that (1 + gamma_1) row(2, 3) = row(0, 1) * (-i, -i), so we constrain row to 0, 1
// only 2 columns have elem, so we constrain col to 0, 1

template <typename _FloatType>
__forceinline__ __device__
Complex<_FloatType> get_scale(int gamma_id, int row, int col) {
    if (gamma_id < 0 || gamma_id > 3) {
        printf("gamma_id out of range\n");
        exit(-1);
    }
    if (!(row == 0 || row == 1) || !(col == 0 || col == 1)) {
        printf("row or col out of range\n");
        exit(-1);
    }
    // 1 + gamma_1 
    if (gamma_id == 0) {
        // if (row == 0) {
        //     if (col == 0) return Complex<_FloatType>(1.0);
        //     if (col == 1) return Complex<_FloatType>(0.0, 1.0);
        // } else {
        //     if (col == 0) return Complex<_FloatType>(1.0);
        //     if (col == 1) return Complex<_FloatType>(0.0, 1.0);
        // }

        // reduce warp divergence
        if (col == 0) return Complex<_FloatType>(1.0);
        if (col == 1) return Complex<_FloatType>(0.0, 1.0);
    }

    else if (gamma_id == 1) {
        // if (row == 0) {
        //     if (col == 0) return Complex<_FloatType>(1.0);
        //     if (col == 1) return Complex<_FloatType>(-1.0);
        // } else {
        //     if (col == 0) return Complex<_FloatType>(1.0);
        //     if (col == 1) return Complex<_FloatType>(1.0);
        // }

        // reduce warp divergence
        if (col == 0) return Complex<_FloatType>(1.0);
        else if (row == 0) return Complex<_FloatType>(-1.0);
        else return Complex<_FloatType>(1.0);
    }

    else if (gamma_id == 2) {
        // if (row == 0) {
        //     if (col == 0) return Complex<_FloatType>(1.0);
        //     if (col == 1) return Complex<_FloatType>(0.0, 1.0);
        // } else {
        //     if (col == 0) return Complex<_FloatType>(1.0);
        //     if (col == 1) return Complex<_FloatType>(0.0, -1.0);
        // }

        // reduce warp divergence
        if (col == 0) return Complex<_FloatType>(1.0);
        else if (row == 0) return Complex<_FloatType>(0.0, 1.0);
        else return Complex<_FloatType>(0.0, -1.0);
    }

    else if (gamma_id == 3) {
        // if (row == 0) {
        //     if (col == 0) return Complex<_FloatType>(1.0);
        //     if (col == 1) return Complex<_FloatType>(1.0);
        // } else {
        //     if (col == 0) return Complex<_FloatType>(1.0);
        //     if (col == 1) return Complex<_FloatType>(1.0);
        // }

        // reduce warp divergence
        return Complex<_FloatType>(1.0);
    }

    // error handling
    printf("gamma_id = %d, row = %d, col = %d, some parameter out of range\n", gamma_id, row, col);
    exit(-1);
}


// always row-major
template <
    typename _FloatType = double,
    int _BLK_M = 16,
    int _BLK_N = 16
>
__forceinline__ __device__ 
void ldg_mat_to_reg(Float2_t<_FloatType>* __restrict__ glb_mat, int M, int N,
                    Float2_t<_FloatType>* __restrict__ reg_mat, int start_m, int start_n) 
{
    int rows = _BLK_M / blockDim.y;  // how many rows each thread load
    int cols = _BLK_N / blockDim.x;  // how many cols each thread load

    int glb_start_m = start_m + threadIdx.y * rows;
    int glb_start_n = start_n + threadIdx.x * cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (glb_start_m + i < M && glb_start_n + j < N) {
                reg_mat[IDX2D(i, j, cols)] = glb_mat[IDX2D(glb_start_m + i, glb_start_n + j, N)];
            } else {
                reg_mat[IDX2D(i, j, cols)] = Float2_t<_FloatType>(0.0);
            }
        }
    }
}


template <
    typename _FloatType = double,
    int _BLK_M = 16,
    int _BLK_N = 16
>
__forceinline__ __device__ 
void sts_mat( Float2_t<_FloatType>* __restrict__ smem_mat,
                     Float2_t<_FloatType>* __restrict__ reg_mat) 
{
    int rows = _BLK_M / blockDim.y;  // how many rows each thread has
    int cols = _BLK_N / blockDim.x;  // how many cols each thread has

    int smem_start_m = threadIdx.y * rows;
    int smem_start_n = threadIdx.x * cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            smem_mat[IDX2D(smem_start_m + i, smem_start_n + j, _BLK_N)] = reg_mat[IDX2D(i, j, cols)];
        }
    }
}


template <
    typename _FloatType = double,
    int _BLK_M = 16,
    int _BLK_N = 16
>
__forceinline__ __device__ 
void sts_mat_transpose( Float2_t<_FloatType>* __restrict__ smem_mat,
                        Float2_t<_FloatType>* __restrict__ reg_mat) 
{
    int rows = _BLK_M / blockDim.y;  // how many rows each thread has
    int cols = _BLK_N / blockDim.x;  // how many cols each thread has

    int smem_start_m = threadIdx.y * rows;
    int smem_start_n = threadIdx.x * cols;
    // transpose
    // 假想一个smem[row][col]到真实的smem[col][row]的transpose
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            smem_mat[IDX2D(smem_start_n + j, smem_start_m + i, _BLK_M)] = reg_mat[IDX2D(i, j, cols)];
        }
    }
}

// assume warp threads is always 4 * 8
template <
    typename _FloatType = double,
    int _BLK_SIZE = 128,
    int _BLK_M = 16,
    int _BLK_N = 16,
    int _BLK_K = 8,
    int _WARP_M = 8,
    int _WARP_N = 8,
    bool _use_tensor_core = false
>
__forceinline__ __device__ 
void single_point_wilson_dslash(
    _FloatType* __restrict__ out, _FloatType* __restrict__ in, _FloatType* __restrict__ gauge,
    // Float2_t<_FloatType>* __restrict__ smem_A, Float2_t<_FloatType>* __restrict__ smem_B,
    int Lx, int Ly, int Lz, int Lt, uint32_t multiprocess , int parity,
    bool dagger_flag, int n_color, int m_rhs, int coord_1dim, 
    _FloatType kappa = 0, bool mat = false) 
{   
    int lane_id, warp_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id) : );
    asm ("mov.u32 %0, %%warpid;" : "=r"(warp_id) : );

    // used for ping pong
    __shared__ Float2_t<_FloatType> smem_A[2][_BLK_M * _BLK_K];
    __shared__ Float2_t<_FloatType> smem_B[2][_BLK_K * _BLK_N];

    // ldg_A and ldg_B are used to load A and B from global memory
    Float2_t<_FloatType> ldg_A[2][_BLK_M * _BLK_K / _BLK_SIZE];
    Float2_t<_FloatType> ldg_B[2][_BLK_K * _BLK_N / _BLK_SIZE];

    // for future utility, now set to 0 
    int row_start = 0;
    int col_start = 0;

    // Batch Gemm , every block calculate one point, 
    // 4 * n_color * m_rhs size of res in register
    // 2 * n_color * m_rhs size of temp_res in register (add to res)
    // in global memory, A: ncolor * ncolor, B: ncolor * m_rhs * 4,
    // in smem, A: ncolor * ncolor, B: ncolor * m_rhs * 2 (combine 2 of 4 in global memory, into 2 in smem)


    // assume we are calculating half volume
    // smem A size = _BLK_M * _BLK_N
    // smem B size = _BLK_K * _BLK_N
    // C size = _BLK_M * _BLK_N in register and Res size is _BLK_N * _BLK_N * 2 in register

    int half_Lx = (Lx << 1);

    Point coord { coord_1dim / (Lz * Ly * half_Lx)
                , coord_1dim % (Lz * Ly * half_Lx) / (Ly * half_Lx)
                , coord_1dim % (Ly * half_Lx) / half_Lx
                , coord_1dim % half_Lx
                , parity};
    Point move_coord;
    // for i = X_DIM, Y_DIM, Z_DIM, T_DIM
    //      for j in BWD, FWD
    //          load A
    //          load B
    //          gemm
    //          add temp{1, 2} to res{1, 2, 3, 4}

    int half_vol = half_Lx * Ly * Lz * Lt;
    Float2_t<_FloatType> scale1;    // when read B, use scale1 B1 + scale2 B2
    Float2_t<_FloatType> scale2;

    int row = row_start + threadIdx.y;
    int col = col_start + threadIdx.x;

    int32_t mat1_pos; // will be 0 or 1, use this to set mat1 position
    int32_t mat2_pos; // will be 2 or 3, use this to set mat2 position     temp_mat = scale1 * mat1 + scale2 * mat2 


    int32_t blocks_m = (n_color + _BLK_M - 1) / _BLK_M;
    int32_t blocks_n = (2 * m_rhs + _BLK_N - 1) / _BLK_N;

    for (int loop_blk_m = 0; loop_blk_m < blocks_m; ++loop_blk_m) {
        for (int loop_blk_n = 0; loop_blk_n < blocks_n; ++loop_blk_n) {

            int row = row_start + loop_blk_m * _WARP_M + threadIdx.y;
            int col = col_start + loop_blk_n * _WARP_N + threadIdx.x;

            if (row < n_color && col < 2 * m_rhs) {
                for (int dim_dir = 0; dim_dir < Nd * DIRECTIONS; dim_dir++) {
                    int dir = dim_dir & 1;  // same with '% DIRECTIONS'
                    int dim = dim_dir >> 1; // same with '/ DIRECTIONS'

                    move_coord = coord.move(dir, dim, Lx, Ly, Lz, Lt);
                    // calculate start addr of global A and B
                    Float2_t<_FloatType>* glb_A = gauge + ((2 * dim + parity) * half_vol +
                                                            IDX4D(coord.T(), coord.Z(), coord.Y(), coord.X(), Lz, Ly, Lx)
                                                            ) * n_color * n_color;
                    Float2_t<_FloatType>* glb_B = in + IDX4D(move_coord.T(), move_coord.Z(), move_coord.Y(), move_coord.X(), Lz, Ly, Lx)
                                                        * n_color * m_rhs * Ns;

                    // set dagger, BE CAREFUL: it is possible to be wrong here
                    if (dir == FWD) { // fwd default: dagger 
                        if (!dagger_flag) {
                            scale2 = -scale2;
                        }
                    } else { // bwd default: not dagger
                        if (dagger_flag) {
                            scale2 = -scale2;
                        }
                    }

                    // get scale
                    if (col < m_rhs) { // col \in [0, m_rhs)
                        scale1 = get_scale<_FloatType>(dim, 0, 0);
                        scale2 = get_scale<_FloatType>(dim, 0, 1);

                        mat1_pos = 0;
                        if (dim == 0 || dim == 1) { mat2_pos = 3; }
                        else { mat2_pos = 2; }
                    } else {            // col \in [m_rhs, 2 * m_rhs)
                        scale1 = get_scale<_FloatType>(dim, 1, 1);
                        scale2 = get_scale<_FloatType>(dim, 1, 1);

                        mat1_pos = 1;
                        if (dim == 2 || dim == 3) { mat2_pos = 3; }
                        else { mat2_pos = 2; }
                    }

                    // main loop
                    for (int k = 0; k < n_color; k += _BLK_K) {
                        // load A from global memory to register, then store to smem
                        if (dir == 0) { // global memory is row-major, col-major in smem
                            ldg_mat_to_reg<_FloatType, _BLK_M, _BLK_K>(glb_A, n_color, n_color, ldg_A[0], row, col); 
                            // when store, transpose
                        } else {        // global memory is col-major, col-major in smem
                        } 


                        // load B from global memory to register
                        // gemm
                        // add to res
                    }
                }
                // store res to global memory
            }

        }
    }



}

// entry function
template <
    typename _FloatType = double,
    int _BLK_M = 16,
    int _BLK_N = 16,
    int _BLK_K = 8,
    int _WARP_M = 8,
    int _WARP_N = 8,
    bool _use_tensor_core = false
>
__global__ void wilson_dslash_su_n_mrhs(
    _FloatType* __restrict__ out, _FloatType* __restrict__ in, _FloatType* __restrict__ gauge,
    int Lx, int Ly, int Lz, int Lt, int g_x, int g_y, int g_z, int g_t, 
    int parity, bool dagger_flag, int n_color, int m_rhs) 
{
    // block切分使用2D，dim3(WARP_SIZE, WARP_NUMBER)
    int block_id = blockIdx.x;
    int grid_size = gridDim.x;  // 1D grid
    int half_vol = Lx * Ly * Lz * Lt / 2;

    // __shared__ _FloatType smem_A[_BLK_M * _BLK_N * 2];
    // __shared__ _FloatType smem_B[_BLK_K * _BLK_N * 2];

    // clang-format off
    for (int i = block_id; i < half_vol; i += grid_size) {
        // single_point_wilson_dslash<_FloatType>(  out, in, gauge, smem, Lx, Ly, Lz, Lt, g_x, g_y, g_z, g_t,
                                            // parity, dagger_flag, n_color, m_rhs, i);
    }
    // clang-format on
}

}  // namespace device
}  // namespace qcu