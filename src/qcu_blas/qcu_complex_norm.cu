#include <stdexcept>
#include <type_traits>
#include <cuda_fp16.h>
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_complex_norm.h"
#include "kernel/reduction/operation.cuh"
#include "kernel/reduction/reduction.cuh"

#include <cublas_v2.h>
#include "qcu_blas_public.h"
#include "check_error/check_cuda.cuh"

#include <mpi.h>
#include "check_error/check_mpi.h"
#include "qcu_config/qcu_config.h"

namespace qcu::qcu_blas {

template <typename OutputFloat, typename InputFloat>
using ComplexNormArgument = typename qcu::qcu_blas::ComplexNorm<OutputFloat, InputFloat>::ComplexNormArgument;


// Norm
template <typename OutputFloat, typename InputFloat>
void ComplexNorm<OutputFloat, InputFloat>::operator()(ComplexNormArgument param) {
    int threads_per_block            = std::min(512, maxThreadsPerBlock);
    int blocks_per_grid              = std::min(div_ceil(param.single_vec_len, threads_per_block), maxGridSize);

    int thread_round1                = threads_per_block;
    int block_round1                 = blocks_per_grid;

    // 1 个 warp打底，不然会出现规约错误
    int thread_round2                = std::min(div_ceil(block_round1, kWarpSize) * kWarpSize, maxThreadsPerBlock);
    int block_round2                 = 1;


    for (int i = 0; i < param.stride; ++i) {
        // first step
        device::reduction::stride_ComplexNorm_step1_kernel <qcu::device::operation::AddOp, OutputFloat, InputFloat>
            <<<block_round1, thread_round1, 0, param.stream>>>
                (reinterpret_cast<OutputFloat*> (param.tmpBuffer), reinterpret_cast<InputFloat*>  (param.input),
                    i, param.stride, param.single_vec_len);
        CHECK_CUDA(cudaGetLastError());
        // second step
        device::reduction::reduceSumStep2_kernel <qcu::device::operation::AddOp, OutputFloat, qcu::device::operation::SqrtOp>
            <<<block_round2, thread_round2, 0, param.stream>>>
                (reinterpret_cast<OutputFloat*>(param.resArr), reinterpret_cast<OutputFloat*>(param.tmpBuffer),
                    i, blocks_per_grid);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(param.stream));
    size_t type_size = 0;
    // MPI_Datatype mpi_datatype;
    if constexpr (std::is_same_v<OutputFloat, double>) {
        type_size = sizeof(double);
    } else if constexpr (std::is_same_v<OutputFloat, float>) {
        type_size = sizeof(float);
    } else {
        throw std::runtime_error("Unsupported type, Output type must be float or double");
    }
    auto mpi_separated_mask = qcu::config::get_mpi_separated_mask();
    if (mpi_separated_mask > 0) {
        OutputFloat local_norm;
        OutputFloat global_norm;
        CHECK_CUDA(cudaMemcpy(&local_norm, param.resArr, type_size, cudaMemcpyDeviceToHost));
        local_norm = local_norm * local_norm;
        CHECK_MPI(MPI_Allreduce(&local_norm, &global_norm, type_size, MPI_BYTE, MPI_SUM, MPI_COMM_WORLD));
        global_norm = sqrt(global_norm);
        CHECK_CUDA(cudaMemcpy(param.resArr, &global_norm, type_size, cudaMemcpyHostToDevice));
    }
}

// instantiation
template struct ComplexNorm<float, float>;
template struct ComplexNorm<double, double>;
template struct ComplexNorm<double, half>;
template struct ComplexNorm<float, half>;

}  // namespace qcu::qcu_blas
