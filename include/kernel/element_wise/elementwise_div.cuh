#pragma once
#include "complex/qcu_complex.cuh"
#include <type_traits>
#include <cuda_fp16.h>

namespace qcu::device::kernel {

template <typename _Tp>
__global__ static   // res = x - ay
void elementwise_div_kernel (_Tp* res, _Tp* x, _Tp* y, int vec_len)
{
  int global_id    = blockDim.x * blockIdx.x + threadIdx.x;
  int total_thread = blockDim.x * gridDim.x;

  _Tp local_res;
  _Tp in_x;
  _Tp in_y;

  for (int i = global_id; i < vec_len; i += total_thread) {
    in_x = x[i]; 
    in_y = y[i];

    local_res = in_x / in_y; 
    res[i] = local_res;
  }
}
}  // namespace qcu::device::kernel