#pragma once

#include <cuda_fp16.h>

namespace qcu {
namespace device {
namespace operation {

template <typename T>
struct UnaryOp {
    virtual __device__ __forceinline__ T operator() (T input) { return input; }   // nothing todo
};

template <typename T>
struct SqrtOp : public UnaryOp <T> {
    __device__ __forceinline__ T operator() (T input) {  return sqrt(input); }
};

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