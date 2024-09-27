#pragma once
#include "complex/qcu_complex.cuh"
// #include "qcu_float_float2_wrapper.h"
#include "base/datatype/qcu_float2.cuh"
#include <type_traits>
#include <cuda_fp16.h>

namespace qcu::device::kernel {

template <typename _Tp, std::enable_if_t <std::is_same_v<_Tp, double> ||
                                          std::is_same_v<_Tp, float>  || 
                                          std::is_same_v<_Tp, half>
                                         >* = nullptr>
__global__ static   // res = ax + by
void caxpby_stride_kernel ( Complex<_Tp>* res, 
                            Complex<_Tp>*  a,   Complex<_Tp>* x, 
                            Complex<_Tp>*  b,   Complex<_Tp>* y, 
                            int single_vec_len, int inc_idx,
                            int start_idx = 0)
{
  using Float2       = typename qcu::Float2_t<_Tp>;
  int   global_id    = blockDim.x * blockIdx.x + threadIdx.x;
  int   total_thread = blockDim.x * gridDim.x;

  Complex<_Tp> in_a = Complex<_Tp>(*reinterpret_cast<Float2*>(a + start_idx));
  Complex<_Tp> in_b = Complex<_Tp>(*reinterpret_cast<Float2*>(b + start_idx));
  Complex<_Tp> in_x;
  Complex<_Tp> in_y;
  Float2       float2_res;
  Complex<_Tp> res_val;
  
  for (int i = global_id; i < single_vec_len; i += total_thread) {
    in_x = Complex<_Tp>(*reinterpret_cast<Float2*>(x + start_idx + i * inc_idx)); 
    in_y = Complex<_Tp>(*reinterpret_cast<Float2*>(y + start_idx + i * inc_idx));

    res_val = in_a * in_x + in_b * in_y;

    float2_res.x = res_val.real();
    float2_res.y = res_val.imag();

    *reinterpret_cast<Float2*>(res + start_idx + i * inc_idx) = float2_res;
  }
}
}  // namespace qcu::device::kernel