#include <stdexcept>

#include "qcu_utils.h"
#include "qcu_blas/qcu_blas.h"
#include "kernel/reduction/operation.cuh"
#include "kernel/reduction/reduction.cuh"

namespace qcu::qcu_blas {



// Norm
template <typename OutputFloat, typename InputFloat>
void ReductionNorm<OutputFloat, InputFloat>::operator()(ReductionParam param) {
  int threads_per_block = 256;
  int blocks_per_grid = div_ceil(param.single_vector_length, threads_per_block);

  for (int i = 0; i < param.m_rhs; ++i) {
    // first step
    device::reduction::strideNorm_1_kernel 
                        <qcu::device::operation::AddOp, OutputFloat, InputFloat> 
                        <<<blocks_per_grid, threads_per_block>>> 
                        ( static_cast<OutputFloat*> (param.tmpBuffer),
                          static_cast<InputFloat*> (param.input1), 
                          i, param.m_rhs, param.single_vector_length);
    CHECK_CUDA(cudaGetLastError());
    // second step
    device::reduction::reduceSumStep2_kernel <qcu::device::operation::AddOp, OutputFloat> 
                          <<<1, blocks_per_grid>>>
                          (param.lastRes, static_cast<OutputFloat*>(param.tmpBuffer), 
                           i, blocks_per_grid);
    CHECK_CUDA(cudaGetLastError());
  }
}

// ReductionInnerprod
template <typename OutputFloat, typename InputFloat>
void ReductionInnerprod<OutputFloat, InputFloat>::operator()(ReductionParam param) {
  int threads_per_block = 256;
  int blocks_per_grid = div_ceil(param.single_vector_length, threads_per_block);

  for (int i = 0; i < param.m_rhs; ++i) {
    device::reduction::strideInnerProd_1_kernel  
                              < qcu::device::operation::AddOp, OutputFloat, InputFloat >
                              <<<blocks_per_grid, threads_per_block>>> 
                              ( static_cast<OutputFloat*> (param.tmpBuffer),
                                static_cast<InputFloat*> (param.input1), 
                                static_cast<InputFloat*> (param.input2), 
                                i, param.m_rhs, param.single_vector_length);
    CHECK_CUDA(cudaGetLastError());
    // second step
    device::reduction::reduceSumStep2_kernel 
                          <qcu::device::operation::AddOp, Complex<OutputFloat>, qcu::device::operation::SqrtOp> // norm2 开根号得出norm
                          <<<1, blocks_per_grid>>>
                          (param.lastRes, static_cast<Complex<OutputFloat>*>(param.tmpBuffer), 
                           i, blocks_per_grid);
    CHECK_CUDA(cudaGetLastError()); 
  }
  
}

}  // namespace qcu::qcu_blas