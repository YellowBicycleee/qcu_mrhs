#pragma once
#include <cuda_fp16.h>
#include <type_traits>

#include "qcu_enum.h"
namespace qcu {

// Float2Wrapper find matching Float2 with Float
template <typename Float = float>
struct Float2Wrapper;

template <>
struct Float2Wrapper<float> {
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

template <typename Float>
using Float2_t = typename Float2Wrapper<Float>::Float2;


// Float2WrapperFromPrecision find matching Float and Float2 with precision
template <QCU_PRECISION precision>
struct Float2WrapperFromPrecision; 

template <>
struct Float2WrapperFromPrecision <QCU_PRECISION::QCU_SINGLE_PRECISION> {
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