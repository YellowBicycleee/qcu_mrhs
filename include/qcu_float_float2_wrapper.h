#pragma once
#include <cuda_fp16.h>
namespace qcu {

template <typename Float = float>
struct Float2Wrapper {
    using Float2 = float2;
    using AccFloat = float;
};
template <>
struct Float2Wrapper<double> {
    using Float2 = double2;
    using AccFloat = double;
};
template <>
struct Float2Wrapper<half> {
    using Float2 = half2;
    using AccFloat = float;
};

}  // namespace qcu