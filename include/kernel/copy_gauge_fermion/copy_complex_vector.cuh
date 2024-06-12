#pragma once
#include <cuda_fp16.h>

#include "qcu_macro.h"

namespace qcu {
namespace device {

// // clang-format off

// complex vector len = len
// float vector len = 2 * len
template <typename Float2_dst, typename Float2_src>
__forceinline__ __device__ __host__ void copyComplexVector(Float2_dst* __restrict__ dst,
                                                           const Float2_src* __restrict__ src, int len) {
    errorQcu("Not implemented\n");
}

__forceinline__ __device__ __host__ void copyComplexVector<float2, half2>(float2* __restrict__ dst,
                                                                          const half2* __restrict__ src, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float2 temp;
    for (int i = idx; i < len; i += stride) {
        // __host__ __device__ float2 __half22float2(const __half2 a)
        temp = __half22float2(src[i]);  // or  temp.x = __half2float(src[i].x); temp.y = __half2float(src[i].y);
        dst[i] = temp;
    }
}
__forceinline__ __device__ __host__ void copyComplexVector<double2, half2>(double2* __restrict__ dst,
                                                                           const half2* __restrict__ src, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    double2 temp;
    half2 src_temp;
    for (int i = idx; i < len; i += stride) {
        src_temp = src[i];
        temp.x = double(src_temp.x);
        temp.y = double(src_temp.y);
        dst[i] = temp;
    }
}

__forceinline__ __device__ __host__ void copyComplexVector<half2, float2>(half2* __restrict__ dst,
                                                                          const float2* __restrict__ src, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    half2 temp;
    for (int i = idx; i < len; i += stride) {
        // __host__ __device__ __half2 __float22half2(const float2 a)
        temp = __float22half2(src[i]);  // or  temp.x = __float2half(src[i].x); temp.y = __float2half(src[i].y);
        dst[i] = temp;
    }
}

__forceinline__ __device__ __host__ void copyComplexVector<half2, double2>(half2* __restrict__ dst,
                                                                           const double2* __restrict__ src, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    half2 temp;
    double2 src_temp;
    for (int i = idx; i < len; i += stride) {
        src_temp = src[i];
        temp.x = __double2half(src_temp.x);
        temp.y = __double2half(src_temp.y);
        dst[i] = temp;
    }
}

__forceinline__ __device__ __host__ void copyComplexVector<float2, double2>(float2* __restrict__ dst,
                                                                            const double2* __restrict__ src, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float2 temp;
    double2 src_temp;
    for (int i = idx; i < len; i += stride) {
        src_temp = src[i];
        temp.x = (float)src_temp.x;
        temp.y = (float)src_temp.y;
        dst[i] = temp;
    }
}

__forceinline__ __device__ __host__ void copyComplexVector<double2, float2>(double2* __restrict__ dst,
                                                                            const float2* __restrict__ src, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    double2 temp;
    float2 src_temp;
    for (int i = idx; i < len; i += stride) {
        src_temp = src[i];
        temp.x = (double)src_temp.x;
        temp.y = (double)src_temp.y;
        dst[i] = temp;
    }
}

}  // namespace device

}  // namespace qcu