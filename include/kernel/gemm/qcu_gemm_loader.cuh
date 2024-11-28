#pragma once

#include <base/datatype/qcu_complex.cuh>
namespace qcu::gemm {

// new function
template <typename Tp_,
    typename BlockShape_ = GemmShape<16, 16, 8>,
    typename WarpShape_ = GemmShape<8, 8, 4>,
    int WarpRow_ = 4,
    int WarpCol = 8
>
QCU_DEVICE
void stg (Tp_* glb, int M, int N, int start_m, int start_n, Tp_* reg) {

    int m = start_m + threadIdx.y;
    int n = start_n + threadIdx.x;

    if (m < M && n < N) {
        glb[m * N + n] = *reg;
    } // else do nothing
}

// new function
template <typename Tp_,
    typename BlockShape_ = GemmShape<16, 16, 8>,
    typename WarpShape_ = GemmShape<8, 8, 4>,
    int WarpRow_ = 4,
    int WarpCol = 8
>
QCU_DEVICE
void ldg (Tp_* glb, int M, int N, int start_m, int start_n, Tp_* reg) {

    int m = start_m + threadIdx.y;
    int n = start_n + threadIdx.x;

    if (m < M && n < N) {
        *reg = glb[m * N + n];
    }
    else { // padding
        *reg = {0};
    }
}


template <typename Float_,
    typename BlockShape_ = GemmShape<16, 16, 8>,
    typename WarpShape_ = GemmShape<8, 8, 4>,
    int WarpRow_ = 4,
    int WarpCol = 8
>
QCU_DEVICE
void ldg_fermion (Float_* glb1, Float_* glb2, int M, int N,
    Complex<Float_> scale, int start_m, int start_n, Float2_t<Float_>* reg)
{

    int m = start_m + threadIdx.y;
    int n = start_n + threadIdx.x;

    Complex<Float_> temp1;
    Complex<Float_> temp2;

    if (m < M && n < N) {
        temp1 = reinterpret_cast<Float2_t<Float_>*>(glb1) [m * N + n];
        temp2 = reinterpret_cast<Float2_t<Float_>*>(glb2) [m * N + n];
        Complex<Float_> temp = temp1 + scale * temp2;
        reg->x = temp.real();
        reg->y = temp.imag();
    }
    else {
        *reg = {0, 0};
    }
}

template <typename Tp_,
    typename BlockShape_ = GemmShape<16, 16, 8>, // K is not used
    typename WarpShape_ = GemmShape<8, 8, 4>,
    int WarpRow_ = 4,
    int WarpCol = 8
>
QCU_DEVICE void sts_direct (Tp_* smem, Tp_* reg) {
    int row = threadIdx.y;
    int col = threadIdx.x ;
    if (row < BlockShape_::kM && col < BlockShape_::kN) {
        smem[row * BlockShape_::kN + col] = * reg;
    }
    // smem[row * BlockShape_::kN + col] = * reg;
    // __syncthreads();
}

template <typename Float_,
    typename BlockShape_ = GemmShape<16, 16, 8>,
    typename WarpShape_ = GemmShape<8, 8, 4>,
    int WarpRow_ = 4,
    int WarpCol = 8
>
QCU_DEVICE void sts_transpose (Float_* smem, Float_* reg) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < BlockShape_::kM && col < BlockShape_::kN) {
        smem[col * BlockShape_::kM + row] = * reg;
    }
    // smem[col * BlockShape_::kM + row] = * reg; // reduce bank conflict
    // __syncthreads();
}


}