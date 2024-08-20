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
using ComplexNormArgument = typename qcu::qcu_blas::ComplexNorm<OutputFloat, InputFloat>::ComplexNormArgument;


// Norm
template <typename OutputFloat, typename InputFloat>
void ComplexNorm<OutputFloat, InputFloat>::operator()(ComplexNormArgument param) {

  constexpr int maxThreadsPerBlock = MAX_THREADS_PER_BLOCK;
  constexpr int maxGridSize        = {2147483647};

  int threads_per_block            = std::min(512, maxThreadsPerBlock);
  int blocks_per_grid              = std::min(div_ceil(param.single_vector_length, threads_per_block),
                                              maxGridSize);

  int thread_round1                = threads_per_block;
  int block_round1                 = blocks_per_grid;
   // 1 个 warp打底，不然会出现规约错误
  int thread_round2                = std::min( div_ceil(block_round1, WARP_SIZE) * WARP_SIZE, 
                                                maxThreadsPerBlock); 
  int block_round2                 = 1;

  // printf("thread_round1 = %d, block_round1 = %d\n", thread_round1, block_round1);
  // printf("thread_round2 = %d, block_round2 = %d\n", thread_round2, block_round2);

  for (int i = 0; i < param.stride; ++i) {
    // first step
    device::reduction::stride_ComplexNorm_step1_kernel
          <qcu::device::operation::AddOp, OutputFloat, InputFloat> 
                          <<<block_round1, thread_round1>>> 
                            ( reinterpret_cast<OutputFloat*> (param.tmpBuffer),
                              reinterpret_cast<InputFloat*>  (param.input), 
                              i, param.stride, param.single_vector_length);
    CHECK_CUDA(cudaGetLastError());
    // second step
    device::reduction::reduceSumStep2_kernel <qcu::device::operation::AddOp, 
                                              OutputFloat,
                                              qcu::device::operation::SqrtOp> 
                          <<<block_round2, thread_round2>>> 
                          ( reinterpret_cast<OutputFloat*>(param.resArr), 
                            reinterpret_cast<OutputFloat*>(param.tmpBuffer), 
                            i, 
                            blocks_per_grid
                          );
    CHECK_CUDA(cudaGetLastError());
  }
}

// instantiation
template struct ComplexNorm<float, float>;
template struct ComplexNorm<double, double>;
template struct ComplexNorm<double, half>;
template struct ComplexNorm<float, half>;

}  // namespace qcu::qcu_blas
