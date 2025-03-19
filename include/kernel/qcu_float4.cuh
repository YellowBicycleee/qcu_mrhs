#pragma once

namespace qcu::Float4 {

template <typename Float_>
struct Float4;

template <>
struct Float4<float> {
    using type = float4;
};

template <>
struct Float4<double> {
    using type = double4;
};

template <>
struct Float4<half> {
    using type = float2;
};

template <typename Float_>
using Float4_t = typename Float4<Float_>::type;
}