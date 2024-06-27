#pragma once

#include "qcu_enum.h"
#include "qcu_float_float2_wrapper.h"
namespace qcu {

void copyComplexVector_interface(void* __restrict__ dst, QCU_PRECISION dstPrec, void* __restrict__ src,
                                 QCU_PRECISION srcPrec, int complex_vector_length, cudaStream_t stream = NULL);

void colorSpinorGather(void* __restrict__ global_dst_ptr, QCU_PRECISION dstPrec, void* __restrict__ global_src_array,
                       QCU_PRECISION srcPrec, int Lx, int Ly, int Lz, int Lt, int n_color, int m_input,
                       cudaStream_t stream = NULL);

void colorSpinorScatter(void* __restrict__ global_dst_array, QCU_PRECISION dstPrec, void* __restrict__ global_src_ptr,
                        QCU_PRECISION srcPrec, int Lx, int Ly, int Lz, int Lt, int n_color, int m_input,
                        cudaStream_t stream = NULL);
}  // namespace qcu