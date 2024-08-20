#pragma once
#include "kernel/shift_data_type.cuh"
#include "kernel/reduction/operation.cuh"
#include "qcu_float_float2_wrapper.h"
#include "qcu_macro.h"
#include <complex/qcu_complex.cuh>
#include <type_traits>

// using namespace qcu::device::operation;

namespace qcu {
namespace device {
namespace reduction {

// template <template <typename> class ReductionOp, 
//           typename Float>
// __device__ __forceinline__ Float warpReduce(Float val) {
//     for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
//         val = ReductionOp<Float>()(val, __shfl_xor_sync(0xffffffff, val, mask));
//     }
//     return val;
// }

template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ T ComplexWarpReduce (T* smem, int lane_id) {
    T val = smem[lane_id];
    T temp;
    ReductionOp <T> op;
    __syncwarp();
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        temp = smem[lane_id ^ mask];
        // __syncwarp();
        val = op(val, temp);
        // val = ReductionOp<T>()(val, temp);
        smem[lane_id] = val;
        // __syncwarp();
    }
    return val;
}

// 把block reduce拆分为多个warp reduce来计算
// T为已经从global memory算完的，目前每个thread只对应一个结果，按道理说不用再区分Output和Input的类型
template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ void blockReduce(T val, T* smem) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid & (WARP_SIZE - 1);
    
    int warp_nums = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;  // 向上进1，以防分配的线程数量小于32导致warp nums为0

    smem[tid] = val;
    __syncwarp();
    val = ComplexWarpReduce<ReductionOp, T>(smem + WARP_SIZE * warp_id, lane_id);

    __syncthreads();
    if (lane_id == 0) {
        smem[warp_id] = val;
    }

    __syncthreads();
    // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
    // 目前的一个假设：block中MAX_THREAD不超过1024，1024/32 = 32，因此这里只看一个WARP
    // T warp_val = tid < warp_nums ? smem[tid] : 0;
    // __syncthreads();
    // smem[tid] = warp_val;
    // __syncthreads();
    if (tid >= warp_nums) {
        smem[tid] = 0;
    }
    __syncthreads();

    if (warp_id == 0) {
        T block_res = ComplexWarpReduce<ReductionOp, T>(smem, lane_id);
        // __syncthreads();
        __syncwarp();
        smem[0] = block_res;
        __syncwarp();
    }

}


// stride norm 
template <template <typename> class ReductionOp, 
          typename OutputFloat,
          typename InputFloat>  // Float
// pos_in_rhs : 表示是mrhs的第几个右手
// m_rhs，也就是m，总共有几个右手
// single_vector_length: 单个右手向量多长 
__global__ void stride_ComplexNorm_step1_kernel (   OutputFloat*       __restrict__ tmpBuffer, 
                                                    const InputFloat * __restrict__ input, 
                                                    int pos_in_rhs, 
                                                    int m_rhs, 
                                                    int single_vector_length) 
{
    
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    __shared__ OutputFloat stride_norm_smem [MAX_THREADS_PER_BLOCK];

    OutputFloat thread_res = 0; // 记录单个线程的结果，OutputType一般为double或者float
    Complex<OutputFloat> tmp;
    ReductionOp<OutputFloat> op; 

    for (int i = global_id; i < single_vector_length; i += total_threads) {                
        tmp = Complex<OutputFloat>(
                    shiftDataType<Float2_t<OutputFloat>, Float2_t<InputFloat>> (
                        reinterpret_cast<const Float2_t<InputFloat>*>(input)[pos_in_rhs + i * m_rhs]
                    ));
        thread_res = op (thread_res, tmp.norm2());
    }
    __syncthreads();


    // reduce block
    blockReduce <ReductionOp, OutputFloat> (thread_res, stride_norm_smem);
    if (0 == threadIdx.x) {
        tmpBuffer [blockIdx.x] = stride_norm_smem[0];
    }
}

// stride innerproduct
template <template <typename> class ReductionOp, 
          typename OutputFloat, 
          typename InputFloat>
__global__ void stride_ComplexInnerProd_step1_kernel (  OutputFloat*  tmpBuffer, 
                                                        const InputFloat *  input1, 
                                                        const InputFloat *  input2, 
                                                        int pos_in_rhs, int m_rhs, int single_vector_length) 
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    __shared__ Complex<OutputFloat> stride_norm_smem [MAX_THREADS_PER_BLOCK];

    Complex<OutputFloat> thread_res = Complex<OutputFloat> (0, 0);
    Complex<OutputFloat> operand1;
    Complex<OutputFloat> operand2;
    ReductionOp<Complex<OutputFloat>> op;

    for (int i = global_id; i < single_vector_length; i += total_threads) {
        operand1 = Complex<OutputFloat> (
                        shiftDataType <Float2_t<OutputFloat>, Float2_t<InputFloat>>(
                            reinterpret_cast<const Float2_t<InputFloat>*>(input1)[pos_in_rhs + i * m_rhs]
                        ));
        operand2 = Complex<OutputFloat> (
                        shiftDataType <Float2_t<OutputFloat>, Float2_t<InputFloat>>(
                            reinterpret_cast<const Float2_t<InputFloat>*>(input2)[pos_in_rhs + i * m_rhs]
                        ));
        
        thread_res = op(thread_res, operand1.conj() * operand2);
    }
    stride_norm_smem[threadIdx.x] = thread_res;
    // __syncthreads();
    __syncwarp();
    
    // reduce block
    blockReduce <ReductionOp, Complex<OutputFloat>> (thread_res, stride_norm_smem);
    __syncthreads();
    if (0 == threadIdx.x) {
        reinterpret_cast<Complex<OutputFloat>*>(tmpBuffer) [blockIdx.x] = stride_norm_smem[0];
        
        // if constexpr (std::is_same_v<OutputFloat, double>) {
        //     printf("blockIdx.x = %d, res = %lf\n", blockIdx.x, stride_norm_smem[0].real());
        // }
    }
}

template <template <typename> class ReductionOp, 
          typename T, 
          template <typename> class RestOp = qcu::device::operation::UnaryOp>
__global__ void reduceSumStep2_kernel (T* output, 
                                       T* tmpBuffer, 
                                       int pos_in_rhs,
                                       int tmp_vec_length) 
{
    __shared__ T smem [MAX_THREADS_PER_BLOCK];

    int global_id    = blockIdx.x * blockDim.x + threadIdx.x;
    int total_thread = gridDim.x  * blockDim.x;

    ReductionOp<T> op;
    T thread_res = {0};
    T tmp;

    for (int i = global_id; i < tmp_vec_length; i += total_thread) {
        tmp = tmpBuffer[i];
        thread_res = op (thread_res, tmp);
    }
    // reduce block
    smem[threadIdx.x] = thread_res;
    __syncthreads();
    blockReduce <ReductionOp, T> (thread_res, smem);
    __syncthreads();
    if (0 == threadIdx.x) {
        // 追加操作默认为空，返回自己
        output[pos_in_rhs] = RestOp<T>()(smem[0]);
    }
}

}  //  nested namespace qcu::device::reduction
}
}
