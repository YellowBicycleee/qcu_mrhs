#pragma once
#include <cuda_fp16.h>

namespace qcu {
namespace device {
template <typename Float = float>
struct WMMA_Param {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
};
template <>
struct WMMA_Param<double> {
    static constexpr int WMMA_M = 8;
    static constexpr int WMMA_N = 8;
    static constexpr int WMMA_K = 4;
};
template <>
struct WMMA_Param<half> {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
};

}  // namespace device
}  // namespace qcu