#pragma once

#include <cuda_fp16.h>

#include "kernel/shift_data_type.cuh"
#include "qcu_public.h"
#include "qcu_utils.h"
namespace qcu {
namespace device {

// clang-format off

// Gather separated FP64 color spinor to FP16 color spinor.
// Float2_dst and Float2_src must be half2, float2, or double2.
// global_src_array must be an device pointer array.
template <typename Float2_dst, typename Float2_src>
__device__ __forceinline__ void gather_color_spinor (Float2_dst*  __restrict__ global_dst_ptr,
                                                     Float2_src** __restrict__ global_src_array,
                                                    int x, int y, int z, int t, 
                                                    int Lx, int Ly, int Lz, int Lt,
                                                    int n_color, int m_input) {
    int half_Lx = Lx / 2;
    int src_offset;
    Float2_dst dst_temp;
    Float2_src src_temp;

    Float2_dst* dst_point_element_ptr = global_dst_ptr + IDX4D(t, z, y, x, Lz, Ly, half_Lx) * (Ns * n_color * m_input);
    Float2_src* src_point_element_ptr;
    src_offset = IDX4D(t, z, y, x, Lz, Ly, half_Lx) * (Ns * n_color);

    for (int i = 0; i < m_input; i++) {
        src_point_element_ptr = global_src_array[i] + src_offset;
        for (int j = 0; j < Ns; j++) {
            for (int k = 0; k < n_color; k++) {
                src_temp = src_point_element_ptr[IDX2D(j, k, n_color)];        // Ns, Nc
                dst_temp = shiftDataType<Float2_dst, Float2_src>(src_temp);
                dst_point_element_ptr[IDX3D(j, k, i, n_color, m_input)] = dst_temp;  // Nc, Ns, m_input
            }
        }
    }
}

template <typename Float2_dst, typename Float2_src>
__device__ __forceinline__ void scatter_color_spinor (Float2_dst**  __restrict__ global_dst_array,
                                                      Float2_src* __restrict__ global_src_ptr,
                                                    int x, int y, int z, int t, 
                                                    int Lx, int Ly, int Lz, int Lt,
                                                    int n_color, int m_input) { 
    int half_Lx = Lx / 2;
    int dst_offset; 
    Float2_dst dst_temp;
    Float2_src src_temp;

    Float2_src* src_point_element_ptr = global_src_ptr + IDX4D(t, z, y, x, Lz, Ly, half_Lx) * (Ns * n_color * m_input);
    Float2_dst* dst_point_element_ptr;
    dst_offset = IDX4D(t, z, y, x, Lz, Ly, half_Lx) * (Ns * n_color);

    for (int i = 0; i < m_input; i++) {
        dst_point_element_ptr = global_dst_array[i] + dst_offset;
        for (int j = 0; j < Ns; j++) {
            for (int k = 0; k < n_color; k++) {
                src_temp = src_point_element_ptr[IDX3D(j, k, i, n_color, m_input)];  // Ns, Nc, m_input
                dst_temp = shiftDataType<Float2_dst, Float2_src>(src_temp);
                dst_point_element_ptr[IDX2D(j, k, n_color)] = dst_temp;  // Ns, Nc
            }
        }
    }
}

template <typename Float2_dst, typename Float2_src>
__global__ void color_spinor_gather_kernel (Float2_dst*  __restrict__ global_dst_ptr,
                                            Float2_src** __restrict__ global_src_array,
                                            int Lx, int Ly, int Lz, int Lt,
                                            int n_color, int m_input) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int half_Lx = Lx / 2;
    int x, y, z, t;
    int vol = half_Lx * Ly * Lz * Lt;

    for (int i = idx; i < vol; i += stride){
        get4DCoord(t, z, y, x, i, Lz, Ly, half_Lx);
        gather_color_spinor<Float2_dst, Float2_src>(global_dst_ptr, global_src_array, x, y, z, t, Lx, Ly, Lz, Lt, n_color, m_input);
    }
}

template <typename Float2_dst, typename Float2_src>
__global__ void color_spinor_scatter_kernel (Float2_dst**  __restrict__ global_dst_array,
                                            Float2_src* __restrict__ global_src_ptr,
                                            int Lx, int Ly, int Lz, int Lt,
                                            int n_color, int m_input) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int half_Lx = Lx / 2;
    int x, y, z, t;
    int vol = half_Lx * Ly * Lz * Lt;

    for (int i = idx; i < vol; i += stride){
        get4DCoord(t, z, y, x, i, Lz, Ly, half_Lx);
        scatter_color_spinor<Float2_dst, Float2_src>(global_dst_array, global_src_ptr, x, y, z, t, Lx, Ly, Lz, Lt, n_color, m_input);
    }
}

}  // namespace device
}  // namespace qcu