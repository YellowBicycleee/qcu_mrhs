#include <stdexcept>
#include <cuda_fp16.h>

#include "check_error/check_cuda.cuh"
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_complex_dotc.h"
#include "kernel/reduction/operation.cuh"
#include "kernel/reduction/reduction.cuh"

#include <cublas_v2.h>
#include "qcu_blas_public.h"

#include <mpi.h>
#include <cmath>
#include "check_error/check_mpi.h"
#include "qcu_config/qcu_config.h"
namespace qcu::qcu_blas {

template <typename OutputFloat, typename InputFloat>
using DotcArgument = typename qcu::qcu_blas::ComplexDotc<OutputFloat, InputFloat>::DotcArgument;

// ReductionInnerprod
template <typename OutputFloat, typename InputFloat>
void ComplexDotc<OutputFloat, InputFloat>::operator()(DotcArgument arg) {

    int threads_per_block            = std::min(512, maxThreadsPerBlock);
    int blocks_per_grid              = std::min(div_ceil(arg.single_vec_len, threads_per_block), maxGridSize);
    int thread_round1                = threads_per_block;
    int block_round1                 = blocks_per_grid;
    // 1 个 warp打底，不然会出现 warp divergence
    int thread_round2 = std::min( div_ceil(block_round1, kWarpSize) * kWarpSize, maxThreadsPerBlock);
    int block_round2  = 1;

    for (int i = 0; i < arg.stride; ++i) {
        device::reduction::stride_ComplexInnerProd_step1_kernel <qcu::device::operation::AddOp, OutputFloat, InputFloat>
            <<<block_round1, thread_round1>>>
                (reinterpret_cast<OutputFloat*> (arg.tmpBuffer), reinterpret_cast<InputFloat*>  (arg.input1),
                reinterpret_cast<InputFloat*>  (arg.input2), i, arg.stride, arg.single_vec_len);
        CHECK_CUDA(cudaGetLastError());
        // second step
        device::reduction::reduceSumStep2_kernel <qcu::device::operation::AddOp, Complex<OutputFloat>, qcu::device::operation::UnaryOp> // norm2 开根号得出norm
            <<<block_round2, thread_round2>>>
                (arg.resArr, arg.tmpBuffer, i, blocks_per_grid);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(arg.stream));

    MPI_Datatype mpi_datatype;
    if constexpr (std::is_same_v<OutputFloat, double>) {
        mpi_datatype = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<OutputFloat, float>) {
        mpi_datatype = MPI_FLOAT;
    } else {
        throw std::runtime_error("Unsupported type, Output type must be float or double");
    }
    auto mpi_separated_mask = qcu::config::get_mpi_separated_mask();
    if (mpi_separated_mask > 0) {
        OutputFloat local_norm[kMaxRHS * 2];
        OutputFloat global_norm[kMaxRHS * 2];
        const int num_rhs = arg.stride;

        CHECK_CUDA(cudaMemcpy(local_norm, arg.resArr, sizeof(OutputFloat) * 2 * num_rhs, cudaMemcpyDeviceToHost));
        CHECK_MPI(MPI_Allreduce(local_norm, global_norm, 2 * num_rhs, mpi_datatype, MPI_SUM, MPI_COMM_WORLD));
        CHECK_CUDA(cudaMemcpy(arg.resArr, global_norm, sizeof(OutputFloat) * 2 * num_rhs, cudaMemcpyHostToDevice));
    }
}

template<>
void ComplexDotc<double, double>::operator() (DotcArgument arg) {
    cublasHandle_t cublas_handle = arg.handle;
    if (nullptr == cublas_handle) {
        throw std::runtime_error("cublas handle is nullptr");
    }
    int stride = arg.stride;
    for (int i = 0; i < stride; ++i) {
        QCU_CHECK_CUBLAS (
            cublasZdotc(cublas_handle, arg.single_vec_len,
                reinterpret_cast<const cuDoubleComplex*>(arg.input1) + i, stride,
                reinterpret_cast<const cuDoubleComplex*>(arg.input2) + i, stride,
                reinterpret_cast<cuDoubleComplex*>(arg.resArr) + i
            ));
    }

    auto mpi_separated_mask = qcu::config::get_mpi_separated_mask();
    if (mpi_separated_mask > 0) {
        double local_norm;
        double global_norm;
        CHECK_CUDA(cudaMemcpy(&local_norm, arg.resArr, sizeof(double) * 2, cudaMemcpyDeviceToHost));
        local_norm = local_norm * local_norm;
        CHECK_MPI(MPI_Allreduce(&local_norm, &global_norm, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
        global_norm = std::sqrt(global_norm);
        CHECK_CUDA(cudaMemcpy(arg.resArr, &global_norm, sizeof(double) * 2, cudaMemcpyHostToDevice));
    }
}
template struct ComplexDotc<double, double>;
template struct ComplexDotc<float, float>;
template struct ComplexDotc<double, half>;
template struct ComplexDotc<float, half>;
}  // namespace qcu::qcu_blas
