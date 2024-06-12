#pragma once

#include <cuda_fp16.h>

namespace qcu {
namespace device {

template <typename _DstType, typename _SrcType>
__forceinline__ __device__ __host__ _DstType shiftDataType(_SrcType src) {
    return static_cast<_DstType>(src);
}

// __half and float
template <>
__forceinline__ __device__ __host__ __half shiftDataType<__half, float>(float src) {
    return __float2half(src);
}
template <>
__forceinline__ __device__ __host__ float shiftDataType<float, __half>(__half src) {
    return __half2float(src);
}

// half2 and float2
template <>
__forceinline__ __device__ __host__ half2 shiftDataType<half2, float2>(float2 src) {
    half2 temp;
    temp.x = __float2half(src.x);
    temp.y = __float2half(src.y);
    return temp;
}
template <>
__forceinline__ __device__ __host__ float2 shiftDataType<float2, half2>(half2 src) {
    float2 temp;
    temp.x = __half2float(src.x);
    temp.y = __half2float(src.y);
    return temp;
}

// __half and double
template <>
__forceinline__ __device__ __host__ __half shiftDataType<__half, double>(double src) {
    return __double2half(src);
}
template <>
__forceinline__ __device__ __host__ double shiftDataType<double, __half>(__half src) {
    return double(src);
}

// half2 and double2
template <>
__forceinline__ __device__ __host__ half2 shiftDataType<half2, double2>(double2 src) {
    half2 temp;
    temp.x = __double2half(src.x);
    temp.y = __double2half(src.y);
    return temp;
}
template <>
__forceinline__ __device__ __host__ double2 shiftDataType<double2, half2>(half2 src) {
    double2 temp;
    temp.x = double(src.x);
    temp.y = double(src.y);
    return temp;
}

// double2 and float2
template <>
__forceinline__ __device__ __host__ double2 shiftDataType<double2, float2>(float2 src) {
    double2 temp;
    temp.x = (double)src.x;
    temp.y = (double)src.y;
    return temp;
}
template <>
__forceinline__ __device__ __host__ float2 shiftDataType<float2, double2>(double2 src) {
    float2 temp;
    temp.x = (float)src.x;
    temp.y = (float)src.y;
    return temp;
}

// half
template <>
__forceinline__ __device__ __host__ half shiftDataType<half, half>(half src) {
    return src;
}
// half2
template <>
__forceinline__ __device__ __host__ half shiftDataType<half2, half2>(half2 src) {
    return src;
}
// float 
template <>
__forceinline__ __device__ __host__ float shiftDataType<float, float>(float src) {
    return src;
}
// float2 
template <>
__forceinline__ __device__ __host__ float2 shiftDataType<float2, float2>(float2 src) {
    return src;
}
// double 
template <>
__forceinline__ __device__ __host__ double shiftDataType<double, double>(double src) {
    return src;
}
// double 
template <>
__forceinline__ __device__ __host__ double2 shiftDataType<double2, double2>(double2 src) {
    return src;
}

}  // namespace device

}  // namespace qcu