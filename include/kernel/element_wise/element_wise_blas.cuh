#pragma once
#include "complex/qcu_complex.cuh"
#include "qcu_float_float2_wrapper.h"

namespace qcu::device::kernel {

template <typename _Float>  // ax
__global__ void Sax_mrhs_kernel ( _Float* res_out,
                                  Complex<_Float> a,  _Float* x_in,
                                  int single_vec_len, int m_rhs                                          
                                ) 
{
  int global_id = blockDim.x * blockIdx.x + threadIdx.x;
  int total_thread = blockDim.x * gridDim.x;

  using Float2 = Float2_t<_Float>;
  Complex<_Float> x;
  Complex<_Float> res = 0;
  Float2 float2_res;
  for (int i = global_id; i < single_vec_len; i += total_thread) {
    x = Complex<_Float> (reinterpret_cast <Float2_t<_Float>*> (x_in) [i]);
    res = a * x;
    float2_res.x = res.real();
    float2_res.y = res.real();
    reinterpret_cast<Float2_t<_Float>*> (res_out) [i] = float2_res;
  }
}

template <typename _Float>  // ax + by
__global__ void Sax_mrhs_kernel (_Float* res_out, 
                                  Complex<_Float> a, Float* x_in,
                                  Complex<_Float> b, Float* y_in,   
                                  int single_vec_len, int m_rhs                                          
                                ) 
{
  int global_id = blockDim.x * blockIdx.x + threadIdx.x;
  int total_thread = blockDim.x * gridDim.x;

  using Float2 = Float2_t<_Float>;

  Complex<_Float> in;
  Complex<_Float> res = 0;
  Float2 float2_res;
  for (int i = global_id; i < single_vec_len; i += total_thread) {
    in = Complex<_Float> (reinterpret_cast <Float2_t<_Float>*> (x_in) [i]); // x
    res = a * x;  // ax
    in = Complex<_Float> (reinterpret_cast <Float2_t<_Float>*> (y_in) [i]); // y
    res += b * y;
    float2_res.x = res.real();
    float2_res.y = res.real();
    reinterpret_cast<Float2_t<_Float>*> (res_out) [i] = float2_res;
  }
}


}