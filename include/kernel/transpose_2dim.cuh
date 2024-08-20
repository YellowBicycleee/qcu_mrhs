#pragma once 
#include <stdio.h>
namespace qcu::device::kernel {


constexpr int M_TILE = 32; 
constexpr int N_TILE = 16;

// constexpr int 

template <typename _Tp>
// transpose input(m * n) to output(n * m)
__global__ void transpose2D_kernel (_Tp* output, const _Tp* input, int m, int n) {

  int block_x_stride = gridDim.x;
  int block_y_stride = gridDim.y;

  int total_logic_blocks_x = (n + blockDim.x - 1) / blockDim.x;
  int total_logic_blocks_y = (m + blockDim.y - 1) / blockDim.y;

  __shared__ _Tp tile[M_TILE][N_TILE + 1];  // erase bank conflict

  for (int i = blockIdx.x; i < total_logic_blocks_x; i += block_x_stride) {
    for (int j = blockIdx.y; j < total_logic_blocks_y; j += block_y_stride) {
      int logic_in_x = i * blockDim.x + threadIdx.x;
      int logic_in_y = j * blockDim.y + threadIdx.y;

      if (logic_in_y < m && logic_in_x < n) {
        tile[threadIdx.y][threadIdx.x] = input[logic_in_y * n + logic_in_x];
      }
      __syncthreads();

      int logic_out_x = j * blockDim.y + threadIdx.x;
      int logic_out_y = i * blockDim.x + threadIdx.y;
  
      if (logic_out_y < n && logic_out_x < m) {
        output[logic_out_y * m + logic_out_x] = tile[threadIdx.x][threadIdx.y];
      }
      __syncthreads();
    }
  }
}

}
