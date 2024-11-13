#pragma once

namespace qcu::gemm {
// always row-major
template <
    typename _FloatType = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8> // only use M, N, donnot use K !!!
>
QCU_DEVICE
void ldg_mat_to_reg(Float2_t<_FloatType>* __restrict__ glb_mat,
                    int start_m, int start_n, int M, int N,
                    Float2_t<_FloatType>* __restrict__ reg_mat)
{
    int rows = BlockShape_::kM / blockDim.y;  // how many rows each thread load
    int cols = BlockShape_::kN / blockDim.x;  // how many cols each thread load

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
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8> // only use M, N, donnot use K !!!
>
QCU_DEVICE
void sts_mat( Float2_t<FloatType_>* __restrict__ smem_mat,
                Float2_t<FloatType_>* __restrict__ reg_mat)
{
    int rows = BlockShape_::kM / blockDim.y;  // how many rows each thread has
    int cols = BlockShape_::kN / blockDim.x;  // how many cols each thread has

    int smem_start_m = threadIdx.y * rows;
    int smem_start_n = threadIdx.x * cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            smem_mat[IDX2D(smem_start_m + i, smem_start_n + j, BlockShape_::kN)] = reg_mat[IDX2D(i, j, cols)];
        }
    }
}


template <
    typename FloatType_ = double,
    typename BlockShape_ = gemm::GemmShape<16, 16, 8> // only use M, N, donnot use K !!!
>
QCU_DEVICE
void sts_mat_transpose( Float2_t<FloatType_>* __restrict__ smem_mat,
                        Float2_t<FloatType_>* __restrict__ reg_mat)
{
    int rows = BlockShape_::kM / blockDim.y;  // how many rows each thread has
    int cols = BlockShape_::kN / blockDim.x;  // how many cols each thread has

    int smem_start_m = threadIdx.y * rows;
    int smem_start_n = threadIdx.x * cols;
    // transpose
    // 假想一个smem[row][col]到真实的smem[col][row]的transpose
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            smem_mat[IDX2D(smem_start_n + j, smem_start_m + i, BlockShape_::kM)] = reg_mat[IDX2D(i, j, cols)];
        }
    }
}
}