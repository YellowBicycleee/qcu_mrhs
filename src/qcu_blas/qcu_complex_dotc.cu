#include <stdexcept>
#include <cuda_fp16.h>
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas.h"
#include "kernel/reduction/operation.cuh"
#include "kernel/reduction/reduction.cuh"

#include <cublas_v2.h>
#include "qcu_blas_public.h"

namespace qcu::qcu_blas {

template <typename OutputFloat, typename InputFloat>
using DotcArgument = typename qcu::qcu_blas::ComplexDotc<OutputFloat, InputFloat>::DotcArgument;

// ReductionInnerprod
template <typename OutputFloat, typename InputFloat>
void ComplexDotc<OutputFloat, InputFloat>::operator()(DotcArgument arg) {


  int threads_per_block            = std::min(512, maxThreadsPerBlock);
  int blocks_per_grid              = std::min(div_ceil(arg.single_vec_len, threads_per_block),
                                              maxGridSize);

  int thread_round1                = threads_per_block;
  int block_round1                 = blocks_per_grid;

  // 1 个 warp打底，不然会出现 warp divergence
  int thread_round2 = std::min( div_ceil(block_round1, WARP_SIZE) * WARP_SIZE, 
                            maxThreadsPerBlock);
  int block_round2  = 1;

  for (int i = 0; i < arg.stride; ++i) {
    device::reduction::stride_ComplexInnerProd_step1_kernel 
            <qcu::device::operation::AddOp, OutputFloat, InputFloat>
                              <<<block_round1, thread_round1>>> 
                              ( reinterpret_cast<OutputFloat*> (arg.tmpBuffer),
                                reinterpret_cast<InputFloat*>  (arg.input1), 
                                reinterpret_cast<InputFloat*>  (arg.input2), 
                                i, arg.stride, arg.single_vec_len
                              );
    CHECK_CUDA(cudaGetLastError());
    // second step
    device::reduction::reduceSumStep2_kernel 
            <qcu::device::operation::AddOp, Complex<OutputFloat>, qcu::device::operation::UnaryOp> // norm2 开根号得出norm
                          <<<block_round2, thread_round2>>>
                          ( arg.resArr, arg.tmpBuffer, i, blocks_per_grid);
    CHECK_CUDA(cudaGetLastError()); 
  }
}

template<>
void ComplexDotc<double, double>::operator() (DotcArgument arg) {
  cublasHandle_t cublas_handle = arg.handle;
  if (nullptr == cublas_handle) {
    throw std::runtime_error("cublas handle is nullptr");
  }
  int stride = arg.stride;
  for (int i = 0; i < stride; ++i) {
    QCU_CHECK_CUBLAS (cublasZdotc(
                        cublas_handle, 
                        arg.single_vec_len, 
                        reinterpret_cast<const cuDoubleComplex*>(arg.input1) + i, stride, 
                        reinterpret_cast<const cuDoubleComplex*>(arg.input2) + i, stride, 
                        reinterpret_cast<cuDoubleComplex*>(arg.resArr) + i
                      ));
  }
}
template struct ComplexDotc<double, double>;
template struct ComplexDotc<float, float>;
template struct ComplexDotc<half, half>;
template struct ComplexDotc<double, half>;
}  // namespace qcu::qcu_blas
