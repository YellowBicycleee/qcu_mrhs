#pragma once

namespace qcu::device {
template <
        typename Float_ = double,
        typename MatrixShape_ = gemm::MatShape<16, 8>,
        typename Float2_ = Float2_t<Float_>
    >   // Float2 in global memory, but Float in shared memory
__forceinline__ __device__ void ldg_and_sts(Float2_* glb, int start_m, int start_n, int M, int N, Float_* smem_real, Float_* smem_imag) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    for (int v_tid = tid; v_tid < MatrixShape_::kMN; v_tid += blockDim.x * blockDim.y) {
        int m = v_tid / MatrixShape_::kN;
        int n = v_tid % MatrixShape_::kN;

        // the idx to load
        int glb_row = start_m + m;
        int glb_col = start_n + n;
        if (glb_row < M && glb_col < N) {
            int glb_idx = glb_row * N + glb_col;
            Float2_ val = glb[glb_idx];
            smem_real[v_tid] = val.x;
            smem_imag[v_tid] = val.y;
        }
        else { // padding with 0
            smem_real[v_tid] = 0;
            smem_imag[v_tid] = 0;
        }
    }
}

template <
        typename Float_ = double,
        typename MatrixShape_ = gemm::MatShape<16, 8>, // the shape of the shared memory,(shape in glb is transposed)
        typename Float2_ = Float2_t<Float_>
    >   // Float2 in global memory, but Float in shared memory
// start_m 和 start_n 是在global memory中的起始位置, M, N为矩阵的大小，这些变量都没有转置
__forceinline__ __device__ void ldg_and_dagger_to_sts(Float2_* glb, int start_m, int start_n, int M, int N, Float_* smem_real, Float_* smem_imag) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    for (int v_tid = tid; v_tid < MatrixShape_::kMN; v_tid += blockDim.x * blockDim.y) {
        int m = v_tid / MatrixShape_::kN;
        int n = v_tid % MatrixShape_::kN;

        // the idx to load
        int glb_row = start_m + n; // transpose
        int glb_col = start_n + m;
        if (glb_row < M && glb_col < N) {
            int glb_idx = glb_row * N + glb_col;
            Float2_ val = glb[glb_idx];
            smem_real[v_tid] = val.x;
            smem_imag[v_tid] = -val.y; // conjugate
        }
        else { // padding with 0
            smem_real[v_tid] = 0;
            smem_imag[v_tid] = 0;
        }
    }
}


template <typename Float_,
    typename MatrixShape_ = gemm::MatShape<16, 8>,
    typename Float2_ = Float2_t<Float_>,
    typename Complex_ = qcu::Complex<Float_>
>
QCU_DEVICE
void ldg_fermion_sts (Float2_* glb1, Float2_* glb2, int start_m, int start_n, int M, int N,
        Complex<Float_> scale, Float_* smem_b_real, Float_* smem_b_imag) {
    {
        int tid = threadIdx.x + threadIdx.y * blockDim.x;

        for (int v_tid = tid; v_tid < MatrixShape_::kMN; v_tid += blockDim.x * blockDim.y) {
            int m = v_tid / MatrixShape_::kN;
            int n = v_tid % MatrixShape_::kN;

            // the idx to load
            int glb_row = start_m + m;
            int glb_col = start_n + n;
            if (glb_row < M && glb_col < N) {
                int glb_idx = glb_row * N + glb_col;
                Complex_ val_1{glb1[glb_idx]};
                Complex_ val_2{glb2[glb_idx]};

                Complex_ temp = val_1 + scale * val_2;
                smem_b_real[v_tid] = temp.real();
                smem_b_imag[v_tid] = temp.imag();
            }
            else { // padding with 0
                smem_b_real[v_tid] = 0;
                smem_b_imag[v_tid] = 0;
            }
        }
    }
}
}