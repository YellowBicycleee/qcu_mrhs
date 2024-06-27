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

template <QCU_PRECISION precision>
struct Float2WrapperFromPrecision {
    using Float = float;
    using Float2 = float2;
};
template <>
struct Float2WrapperFromPrecision<QCU_PRECISION::QCU_DOUBLE_PRECISION> {
    using Float = double;
    using Float2 = double2;
};
template <>
struct Float2WrapperFromPrecision<QCU_PRECISION::QCU_HALF_PRECISION> {
    using Float = half;
    using Float2 = half2;
};

}  // namespace qcu