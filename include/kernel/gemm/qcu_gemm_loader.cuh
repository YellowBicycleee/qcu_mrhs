#pragma once

#include <complex/qcu_complex.cuh>
namespace qcu::gemm {

// new function
template <typename Tp_,
    typename MatShape_ = MatShape<16, 8>,
    typename BlockShape_ = GemmShape<16, 16, 8>,
    typename WarpShape_ = GemmShape<8, 8, 4>,
    int WarpRow_ = 4,
    int WarpCol = 8
>
QCU_DEVICE
void stg (Tp_* glb, int M, int N, int start_m, int start_n, Tp_* reg) {

    int m = start_m + threadIdx.y;
    int n = start_n + threadIdx.x;

    // if (m < M && n < N) {
    //     glb[m * N + n] = *reg;
    // } // else do nothing
    if (m < M && n < N && threadIdx.y < MatShape_::kM && threadIdx.x < MatShape_::kN) {
        glb[m * N + n] = *reg;
    } // else do nothing
}

// new function
template <typename Tp_,
    typename MatShape_ = MatShape<16, 8>,
    typename BlockShape_ = GemmShape<16, 16, 8>,
    typename WarpShape_ = GemmShape<8, 8, 4>,
    int WarpRow_ = 4,
    int WarpCol = 8
>
QCU_DEVICE
void ldg (Tp_* glb, int M, int N, int start_m, int start_n, Tp_* reg) {

    int m = start_m + threadIdx.y;
    int n = start_n + threadIdx.x;

    // if (m < M && n < N) {
    if (m < M && n < N && threadIdx.y < MatShape_::kM && threadIdx.x < MatShape_::kN) {
        *reg = glb[m * N + n];
    }
    else { // padding
        *reg = {0};
    }
}


template <typename Float_,
    typename MatShape_ = MatShape<16, 8>,
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

    // if (m < M && n < N) {
    if (m < M && n < N && threadIdx.y < MatShape_::kM && threadIdx.x < MatShape_::kN) {
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
    typename MatShape_ = MatShape<16, 8>,
    typename BlockShape_ = GemmShape<16, 16, 8>, // K is not used
    typename WarpShape_ = GemmShape<8, 8, 4>
>
QCU_DEVICE void sts_direct (Tp_* smem, Tp_* reg) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < MatShape_::kM && col < MatShape_::kN) {
        smem[row * MatShape_::kN + col] = * reg;
    }
}

template <
    typename Float_,
    typename MatShape_ = MatShape<16, 8>,
    typename BlockShape_ = GemmShape<16, 16, 8>, // K is not used
    typename WarpShape_ = GemmShape<8, 8, 4>,
    typename Complex_ = qcu::Complex<Float_>
>
QCU_DEVICE void sts_direct (Float_* smem_r, Float_* smem_i, Complex_* reg) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < MatShape_::kM && col < MatShape_::kN) {
        smem_r[row * MatShape_::kN + col] = reg->real();
        smem_i[row * MatShape_::kN + col] = reg->imag();
    }
}

template <typename Float_,
    typename MatShape_ = MatShape<16, 8>,
    typename BlockShape_ = GemmShape<16, 16, 8>,
    typename WarpShape_ = GemmShape<8, 8, 4>
>
QCU_DEVICE void sts_transpose (Float_* smem, Float_* reg) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < MatShape_::kM && col < MatShape_::kN) {
        smem[col * MatShape_::kM + row] = * reg;
    }
}

template <typename Float_,
    typename MatShape_ = MatShape<16, 8>,
    typename BlockShape_ = GemmShape<16, 16, 8>,
    typename WarpShape_ = GemmShape<8, 8, 4>,
    typename Complex_ = qcu::Complex<Float_>
>
QCU_DEVICE void sts_transpose (Float_* smem_r, Float_* smem_i, Float_* reg) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < MatShape_::kM && col < MatShape_::kN) {
        // smem[col * MatShape_::kM + row] = * reg;
        smem_r[col * MatShape_::kM + row] = reg->real();
        smem_i[col * MatShape_::kM + row] = reg->imag();
    }
}


}