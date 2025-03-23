#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mpi.h>
#include <stdexcept>
#include <type_traits>

#include "check_error/check_cuda.cuh"
#include "check_error/check_mpi.h"
#include "kernel/reduction/operation.cuh"
#include "kernel/reduction/reduction.cuh"
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_complex_norm.h"
#include "qcu_blas_public.h"
#include "qcu_config/qcu_config.h"

namespace qcu::qcu_blas {

template <typename OutputFloat, typename InputFloat>
using ComplexNormArgument = typename qcu::qcu_blas::ComplexNorm<OutputFloat, InputFloat>::ComplexNormArgument;


// Norm
template <typename OutputFloat, typename InputFloat>
void ComplexNorm<OutputFloat, InputFloat>::operator()(ComplexNormArgument param) {
    if constexpr(std::is_same_v<InputFloat, float>) {
        for (int i = 0; i < param.stride; ++i) {
            QCU_CHECK_CUBLAS (
                cublasScnrm2(param.handle,
                            param.single_vec_len,
                            reinterpret_cast<cuComplex*>(param.input) + i,
                            param.stride,
                            reinterpret_cast<float*>(param.resArr) + i)
            );
        }
    }
    else if constexpr(std::is_same_v<InputFloat, double>) {
        for (int i = 0; i < param.stride; ++i) {
            QCU_CHECK_CUBLAS (
                cublasDznrm2(param.handle,
                            param.single_vec_len,
                            reinterpret_cast<cuDoubleComplex*>(param.input) + i,
                            param.stride,
                            reinterpret_cast<double*>(param.resArr) + i)
            );
        }
    }
    else {
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
    }
    CHECK_CUDA(cudaStreamSynchronize(param.stream));
    
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
        int num_rhs = param.stride;
        OutputFloat local_norm[kMaxRHS];
        OutputFloat global_norm[kMaxRHS];

        CHECK_CUDA(cudaMemcpy(local_norm, param.resArr, sizeof(OutputFloat) * num_rhs, cudaMemcpyDeviceToHost));
        
        #pragma omp parallel for
        for (int i = 0; i < num_rhs; ++i) {
            local_norm[i] = local_norm[i] * local_norm[i];
        }
        CHECK_MPI(MPI_Allreduce(local_norm, global_norm, num_rhs, mpi_datatype, MPI_SUM, MPI_COMM_WORLD));
        #pragma omp parallel for
        for (int i = 0; i < num_rhs; ++i) {
            global_norm[i] = std::sqrt(global_norm[i]);
        }
        CHECK_CUDA(cudaMemcpy(param.resArr, global_norm, sizeof(OutputFloat) * num_rhs, cudaMemcpyHostToDevice));
    }
}

// instantiation
template struct ComplexNorm<float, float>;
template struct ComplexNorm<double, double>;
template struct ComplexNorm<double, half>;
template struct ComplexNorm<float, half>;

}  // namespace qcu::qcu_blas
