#pragma once
#include "kernel/copy_complex_vector.cuh"

namespace qcu {
namespace device {
template <typename Float2_dst, typename Float2_src>
__global__ void copyGauge(Float2_dst* __restrict__ dst, const Float2_src* __restrict__ src, int len) {
    copyComplexVector(dst, src, len);
}
}  // namespace device
}  // namespace qcu