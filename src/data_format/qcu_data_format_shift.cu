#include <cstdio>

#include "data_format/qcu_data_format_shift.cuh"
#include "kernel/copy_vector/copy_color_spinor.cuh"
#include "kernel/copy_vector/copy_complex_vector.cuh"
#include "qcu_public.h"
#include "check_error/check_cuda.cuh"
#include "base/datatype/qcu_float2.cuh"
namespace qcu {

template <typename DestFloat, typename SrcFloat>
static void copyVector_Complex(void* __restrict__ dst, void* __restrict__ src, int complex_vector_length,
                        cudaStream_t stream) {
    using DestFloat2 = typename qcu::Float2Wrapper<DestFloat>::Float2;
    using SrcFloat2 = typename qcu::Float2Wrapper<SrcFloat>::Float2;
    int block_size = 256;
    int grid_size = complex_vector_length / block_size;
    device::copyComplexVector<DestFloat2, SrcFloat2>
        <<<grid_size, block_size>>>(static_cast<DestFloat2*>(dst), static_cast<SrcFloat2*>(src), complex_vector_length);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename DestFloat, typename SrcFloat>
static void copyVector_Complex_Async(void* __restrict__ dst, void* __restrict__ src, int complex_vector_length,
                              cudaStream_t stream) {
    using DestFloat2 = typename qcu::Float2Wrapper<DestFloat>::Float2;
    using SrcFloat2 = typename qcu::Float2Wrapper<SrcFloat>::Float2;
    int block_size = 256;
    int grid_size = complex_vector_length / block_size;
    device::copyComplexVector<DestFloat2, SrcFloat2>
        <<<grid_size, block_size, 0, stream>>>(static_cast<DestFloat2*>(dst), static_cast<SrcFloat2*>(src), complex_vector_length);
    CHECK_CUDA(cudaGetLastError());
}

template <typename DstFloat, typename SrcFloat>
void colorSpinorScatter(void* __restrict__ global_dst_array, void* __restrict__ global_src_ptr, int Lx, int Ly, int Lz,
                        int Lt, int n_color, int m_input, cudaStream_t stream) {
    using DstFloat2 = typename qcu::Float2Wrapper<DstFloat>::Float2;
    using SrcFloat2 = typename qcu::Float2Wrapper<SrcFloat>::Float2;
    int block_size = 256;
    int grid_size = (Lx / 2 * Ly * Lz * Lt + block_size - 1) / block_size;
    device::color_spinor_scatter_kernel<DstFloat2, SrcFloat2>
        <<<grid_size, block_size, 0, stream>>>(static_cast<DstFloat2**>(global_dst_array), static_cast<SrcFloat2*>(global_src_ptr),
                                    Lx, Ly, Lz, Lt, n_color, m_input);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename DstFloat, typename SrcFloat>
void colorSpinorGather(void* __restrict__ global_dst_ptr, void* __restrict__ global_src_array, int Lx, int Ly, int Lz,
                       int Lt, int n_color, int m_input, cudaStream_t stream) {
    using DstFloat2 = typename qcu::Float2Wrapper<DstFloat>::Float2;
    using SrcFloat2 = typename qcu::Float2Wrapper<SrcFloat>::Float2;
    int block_size = 256;
    int grid_size = (Lx * Ly * Lz * Lt / 2 + block_size - 1) / block_size;
    // printf("DEBUG file %s, line %d, global_src_array = %p, global_dst_ptr = %p\n", __FILE__, __LINE__,global_src_array, global_dst_ptr);
    device::color_spinor_gather_kernel<DstFloat2, SrcFloat2>
        <<<grid_size, block_size>>>(static_cast<DstFloat2*>(global_dst_ptr), static_cast<SrcFloat2**>(global_src_array),
                                    Lx, Ly, Lz, Lt, n_color, m_input);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename DstFloat>
static void instantiate_copyVector_Complex_SrcFloat(void* __restrict__ dest, void* __restrict__ src,
                                                    QcuPrecision srcPrec, int complex_vector_length,
                                                    cudaStream_t stream) {
    switch (srcPrec) {
        case QcuPrecision::kPrecisionHalf:
            copyVector_Complex<DstFloat, half>(dest, src, complex_vector_length, stream);
            break;
        case QcuPrecision::kPrecisionSingle:
            copyVector_Complex<DstFloat, float>(dest, src, complex_vector_length, stream);
            break;
        case QcuPrecision::kPrecisionDouble:
            copyVector_Complex<DstFloat, double>(dest, src, complex_vector_length, stream);
            break;
        default:
            errorQcu("Unsupported Source Float precision\n");
    };
}

template <typename DstFloat>
static void instantiate_colorSpinorScatter_SrcFloat(void* __restrict__ global_dst_array,
                                                    void* __restrict__ global_src_ptr,
                                                    QcuPrecision srcPrec, int Lx,
                                                    int Ly, int Lz, int Lt, int n_color, int m_input,
                                                    cudaStream_t stream) {
    switch (srcPrec) {
        case QcuPrecision::kPrecisionHalf:
            colorSpinorScatter<DstFloat, half>(global_dst_array, global_src_ptr, Lx, Ly, Lz, Lt, n_color, m_input,
                                               stream);
            break;
        case QcuPrecision::kPrecisionSingle:
            colorSpinorScatter<DstFloat, float>(global_dst_array, global_src_ptr, Lx, Ly, Lz, Lt, n_color, m_input,
                                                stream);
            break;
        case QcuPrecision::kPrecisionDouble:
            colorSpinorScatter<DstFloat, double>(global_dst_array, global_src_ptr, Lx, Ly, Lz, Lt, n_color, m_input,
                                                 stream);
            break;
        default:
            errorQcu("Unsupported Source Float precision\n");
    };
}

template <typename DstFloat>
static void instantiate_colorSpinorGather_SrcFloat(void* __restrict__ global_dst_ptr,
                                                   void* __restrict__ global_src_array, QcuPrecision srcPrec, int Lx,
                                                   int Ly, int Lz, int Lt, int n_color, int m_input,
                                                   cudaStream_t stream) {
    switch (srcPrec) {
        case QcuPrecision::kPrecisionHalf:
            colorSpinorGather<DstFloat, half>(global_dst_ptr, global_src_array, Lx, Ly, Lz, Lt, n_color, m_input,
                                              stream);
            break;
        case QcuPrecision::kPrecisionSingle:
            colorSpinorGather<DstFloat, float>(global_dst_ptr, global_src_array, Lx, Ly, Lz, Lt, n_color, m_input,
                                               stream);
            break;
        case QcuPrecision::kPrecisionDouble:
            colorSpinorGather<DstFloat, double>(global_dst_ptr, global_src_array, Lx, Ly, Lz, Lt, n_color, m_input,
                                                stream);
            break;
        default:
            errorQcu("Unsupported Destination Float precision\n");
    }
}

void copyComplexVector_interface(void* __restrict__ dest, QcuPrecision destPrec, void* __restrict__ src,
                                 QcuPrecision srcPrec, int complex_vector_length, cudaStream_t stream) {
    if (destPrec == QcuPrecision::kPrecisionUndefined || srcPrec == QcuPrecision::kPrecisionUndefined) {
        errorQcu("Undefined precision\n");
    }
    // instantiate the template function
    switch (destPrec) {
        case QcuPrecision::kPrecisionHalf:
            instantiate_copyVector_Complex_SrcFloat<half>(dest, src, srcPrec, complex_vector_length, stream);
            break;
        case QcuPrecision::kPrecisionSingle:
            instantiate_copyVector_Complex_SrcFloat<float>(dest, src, srcPrec, complex_vector_length, stream);
            break;
        case QcuPrecision::kPrecisionDouble:
            instantiate_copyVector_Complex_SrcFloat<double>(dest, src, srcPrec, complex_vector_length, stream);
            break;
        default:
            errorQcu("Unsupported Destination Float precision\n");
            break;
    }
}

void colorSpinorScatter(void* __restrict__ global_dst_array, QcuPrecision dstPrec, void* __restrict__ global_src_ptr,
                        QcuPrecision srcPrec, int Lx, int Ly, int Lz, int Lt, int n_color, int m_input,
                        cudaStream_t stream) {
    switch (dstPrec) {
        case QcuPrecision::kPrecisionHalf:
            instantiate_colorSpinorScatter_SrcFloat<half>(global_dst_array, global_src_ptr, srcPrec, Lx, Ly, Lz, Lt,
                                                          n_color, m_input, stream);
            break;
        case QcuPrecision::kPrecisionSingle:
            instantiate_colorSpinorScatter_SrcFloat<float>(global_dst_array, global_src_ptr, srcPrec, Lx, Ly, Lz, Lt,
                                                           n_color, m_input, stream);
            break;
        case QcuPrecision::kPrecisionDouble:
            instantiate_colorSpinorScatter_SrcFloat<double>(global_dst_array, global_src_ptr, srcPrec, Lx, Ly, Lz, Lt,
                                                            n_color, m_input, stream);
            break;
        default:
            errorQcu("Unsupported Destination Float precision\n");
            break;
    }
}

void colorSpinorGather(void* __restrict__ global_dst_ptr, QcuPrecision dstPrec, void* __restrict__ global_src_array,
                       QcuPrecision srcPrec, int Lx, int Ly, int Lz, int Lt, int n_color, int m_input,
                       cudaStream_t stream) {
    switch (dstPrec) {
        case QcuPrecision::kPrecisionHalf:
            instantiate_colorSpinorGather_SrcFloat<half>(global_dst_ptr, global_src_array, srcPrec, Lx, Ly, Lz, Lt,
                                                         n_color, m_input, stream);
            break;
        case QcuPrecision::kPrecisionSingle:
            instantiate_colorSpinorGather_SrcFloat<float>(global_dst_ptr, global_src_array, srcPrec, Lx, Ly, Lz, Lt,
                                                          n_color, m_input, stream);
            break;
        case QcuPrecision::kPrecisionDouble:
            instantiate_colorSpinorGather_SrcFloat<double>(global_dst_ptr, global_src_array, srcPrec, Lx, Ly, Lz, Lt,
                                                           n_color, m_input, stream);
            break;
        default:
            errorQcu("Unsupported Destination Float precision\n");
            break;
    }
}
}  // namespace qcu
