#include "kernel/transpose_2dim.cuh"
#include "qcu_blas/qcu_transpose_2d.h"
#include "qcu_utils.h"
#include "qcu_public.h"

#include "check_error/check_cuda.cuh"
#include "complex/qcu_complex.cuh"
#include <cuda_fp16.h>
#include <iostream>
// #include <cuda_device_runtime_api.h>
namespace qcu::qcu_blas {
template 
void Transpose2D<double2>::operator () ();

template
void Transpose2D<float2>::operator () ();

template
void Transpose2D<half2>::operator () ();

template
void Transpose2D<double>::operator () ();

template
void Transpose2D<float>::operator () ();

template
void Transpose2D<int>::operator () ();

template <typename _Tp>
void Transpose2D<_Tp>::operator () () {

    int max_block_x = 1 << 16; // prop.maxGridSize[0];// .maxThreadsPerBlock;
    int max_block_y = (1 << 16) - 1; //prop.maxGridSize[0];// .maxThreadsPerBlock;
    std::cout << "max_block_x = " << max_block_x << std::endl;
    std::cout << "max_block_y = " << max_block_y << std::endl;

    int threads_per_block_x = qcu::device::kernel::N_TILE;
    int threads_per_block_y = qcu::device::kernel::M_TILE;
    int blocks_per_grid_y = std::min(div_ceil(param.m, threads_per_block_y), max_block_y);
    int blocks_per_grid_x = std::min(div_ceil(param.n, threads_per_block_x), max_block_x);

    printf("block_size = %d %d\n", threads_per_block_x, threads_per_block_y);
    printf("grid_size = %d %d\n", blocks_per_grid_x, blocks_per_grid_y);
    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

    device::kernel::transpose2D_kernel<_Tp> <<< grid_size, block_size >>>
        (static_cast<_Tp*> (param.output), static_cast<_Tp*> (param.input), param.m, param.n);
    CHECK_CUDA(cudaGetLastError());
}


}