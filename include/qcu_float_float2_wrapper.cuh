#pragma once
#include <cuda_fp16.h>
namespace qcu {

template <typename Float = float>
struct {
    using Float2 = float2;
};
template <>
struct<double> {
    using Float2 = double2;
};
template <>
struct<half> {
    using Float2 = half2;
};

}