#pragma once
#include "complex/qcu_complex.cuh"
// #include "qcu_float_float2_wrapper.h"
#include <cuda_fp16.h>

#include <kernel/shift_data_type.cuh>
#include <type_traits>

#include "base/datatype/qcu_float2.cuh"

namespace qcu::device::kernel {
// scale 精度更高
template <
    typename ComputeFloat_,
    typename ScaleFloat_ = ComputeFloat_,
    std::enable_if_t <std::is_same_v<ComputeFloat_, double> || std::is_same_v<ComputeFloat_, float> || std::is_same_v<ComputeFloat_, half>>* = nullptr
>
__global__ static   // res = ax + by
void caxpby_stride_kernel (
    Complex<ComputeFloat_>* res,
    Complex<ComputeFloat_>*  a,
    Complex<ComputeFloat_>* x,
    Complex<ComputeFloat_>*  b,
    Complex<ComputeFloat_>* y,
    int single_vec_len,
    int inc_idx,
    int start_idx = 0)
{
    using ComputeFloat2 = typename qcu::Float2_t<ComputeFloat_>;
    using ScaleFloat2   = typename qcu::Float2_t<ScaleFloat_>;
    int   global_id     = blockDim.x * blockIdx.x + threadIdx.x;
    int   total_thread  = blockDim.x * gridDim.x;

    // load scale
    Complex<ScaleFloat_> in_a = Complex<ScaleFloat_>(*reinterpret_cast<ScaleFloat2*>(a + start_idx));
    Complex<ScaleFloat_> in_b = Complex<ScaleFloat_>(*reinterpret_cast<ScaleFloat2*>(b + start_idx));

    ComputeFloat2 in_x;
    ComputeFloat2 in_y;
    ComputeFloat2 float2_res;
    ComputeFloat2 res_val;

    for (int i = global_id; i < single_vec_len; i += total_thread) {
        in_x = *reinterpret_cast<ComputeFloat2*>(x + start_idx + i * inc_idx);
        in_y = *reinterpret_cast<ComputeFloat2*>(y + start_idx + i * inc_idx);

        Complex<ScaleFloat_> high_precision_in_x = shiftDataType<ScaleFloat2, ComputeFloat2>(in_x);
        Complex<ScaleFloat_> high_precision_in_y = shiftDataType<ScaleFloat2, ComputeFloat2>(in_y);
        Complex<ScaleFloat_> high_precision_res_val
            = in_a * high_precision_in_x + in_b * high_precision_in_y;

        res_val.x = shiftDataType<ComputeFloat_, ScaleFloat_>(high_precision_res_val.real());
        res_val.y = shiftDataType<ComputeFloat_, ScaleFloat_>(high_precision_res_val.imag());
        *reinterpret_cast<ComputeFloat2*>(res + start_idx + i * inc_idx) = res_val;
    }
}
}  // namespace qcu::device::kernel