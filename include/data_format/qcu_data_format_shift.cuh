#pragma once

#include "qcu_public.h"
namespace qcu {

void copyComplexVector_interface(void* __restrict__ dst, QcuPrecision dstPrec, void* __restrict__ src,
                                 QcuPrecision srcPrec, int complex_vector_length, cudaStream_t stream = NULL);

void colorSpinorGather(void* __restrict__ global_dst_ptr, QcuPrecision dstPrec, void* __restrict__ global_src_array,
                       QcuPrecision srcPrec, int Lx, int Ly, int Lz, int Lt, int n_color, int m_input,
                       cudaStream_t stream = NULL);

void colorSpinorScatter(void* __restrict__ global_dst_array, QcuPrecision dstPrec, void* __restrict__ global_src_ptr,
                        QcuPrecision srcPrec, int Lx, int Ly, int Lz, int Lt, int n_color, int m_input,
                        cudaStream_t stream = NULL);
}  // namespace qcu