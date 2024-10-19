#pragma once
#include "complex/qcu_complex.cuh"
#include <type_traits>
#include <cuda_fp16.h>

namespace qcu::device::kernel {

template <typename _Tp>
__global__ static   // res = x - ay
void elementwise_init_kernel (_Tp* res, _Tp scala, int vec_len)
{
  int global_id    = blockDim.x * blockIdx.x + threadIdx.x;
  int total_thread = blockDim.x * gridDim.x;


  for (int i = global_id; i < vec_len; i += total_thread) {
    res[i] = scala;
  }
}

}  // namespace qcu::device::kernel