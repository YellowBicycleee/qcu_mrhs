#pragma once

#include <cuda_fp16.h>

namespace qcu {
namespace devide {
namespace operation {

template <typename T>
struct AddOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct SquareOp {
    __device__ __forceinline__ T operator()(const T& a) const { return a * a; }
};

}  // namespace operation
}
}