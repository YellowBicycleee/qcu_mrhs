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

template <template <typename> class ReductionOp, 
          typename Float>
__device__ __forceinline__ Float warpReduce(Float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val = ReductionOp<Float>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// 把block reduce拆分为多个warp reduce来计算
// T为已经从global memory算完的，目前每个thread只对应一个结果，按道理说不用再区分Output和Input的类型
template <template <typename> class ReductionOp, 
          typename Float>
__device__ __forceinline__ void blockReduce(Float val, Float* smem) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid & (WARP_SIZE - 1);
    
    int warp_nums = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;  // 向上进1，以防分配的线程数量小于32导致warp nums为0

    val = warpReduce<ReductionOp, Float>(val); // 先warp reduce

    if (lane_id == 0) { // TODO: 这个条件可能可以去掉
        smem[warp_id] = val;
    }
    __syncthreads();
    // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
    // 目前的一个假设：block中MAX_THREAD不超过1024，1024/32 = 32，因此这里只看一个WARP
    Float warp_val = tid < warp_nums ? smem[tid] : 0;
    Float block_res = warpReduce<ReductionOp, Float>(warp_val);
    if (tid == 0) {
        smem[0] = block_res;
    }
}


// stride norm 
template <template <typename> class ReductionOp, 
          typename OutputFloat,
          typename InputFloat>  // Float
// pos_in_rhs : 表示是mrhs的第几个右手
// m_rhs，也就是m，总共有几个右手
// single_vector_length: 单个右手向量多长 
__global__ void stride_ComplexNorm_step1_kernel (   OutputFloat* __restrict__ tmpBuffer, 
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

    for (int i = 0; i < single_vector_length; i += total_threads) {                
        tmp = Complex<Float2_t<OutputFloat>>(
                    shiftDataType<Float2_t<OutputFloat>, Float2_t<InputFloat>> (
                        reinterpret_cast<Float2_t<InputFloat>*>(input)[pos_in_rhs + i * m_rhs]
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
__global__ void stride_ComplexInnerProd_step1_kernel (  OutputFloat* __restrict__ tmpBuffer, 
                                                        const InputFloat * __restrict__ input1, 
                                                        const InputFloat * __restrict__ input2, 
                                                        int pos_in_rhs, int m_rhs, int single_vector_length) 
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    __shared__ Complex<OutputFloat> stride_norm_smem [MAX_THREADS_PER_BLOCK];

    Complex<OutputFloat> thread_res = 0;
    Complex<OutputFloat> operand1;
    Complex<OutputFloat> operand2;
    ReductionOp<Complex<OutputFloat>> op;

    for (int i = 0; i < single_vector_length; i += total_threads) {
        operand1 = Complex<OutputFloat> (
                        shiftDataType <Float2_t<OutputFloat>, Float2_t<InputFloat>>(
                            reinterpret_cast<Float2_t<InputFloat>*>(input1)[pos_in_rhs + i * m_rhs]
                        ));
        operand1 = Complex<OutputFloat> (
                        shiftDataType <Float2_t<OutputFloat>, Float2_t<InputFloat>>(
                            reinterpret_cast<Float2_t<InputFloat>*>(input2)[pos_in_rhs + i * m_rhs]
                        ));
        thread_res = op(thread_res, operand1.conj() * operand2);
    }
    // reduce block
    blockReduce <ReductionOp, Complex<OutputFloat>> (thread_res, stride_norm_smem);
    if (0 == threadIdx.x) {
        reinterpret_cast<Complex<OutputFloat>*>(tmpBuffer) [blockIdx.x] = stride_norm_smem[0];
    }
}

template <template <typename> class ReductionOp, 
          typename Float, 
          template <typename> class RestOp = qcu::device::operation::UnaryOp>
__global__ void reduceSumStep2_kernel (Float** output, 
                                       Float* tmpBuffer, 
                                       int pos_in_rhs, 
                                       int tmp_vec_length) 
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_thread = gridDim.x * blockDim.x;

    __shared__ Float smem [MAX_THREADS_PER_BLOCK];
    ReductionOp<Float> op;
    Float thread_res = 0;
    Float tmp;
    for (int i = global_id; i < tmp_vec_length; i += total_thread) {
        tmp = reinterpret_cast<Float*>(tmpBuffer)[i];
        thread_res = op (thread_res, tmp);
    }
    // reduce block
    blockReduce <ReductionOp, Float> (thread_res, smem);
    if (0 == threadIdx.x) {
        // 追加操作默认为空，返回自己
        reinterpret_cast<Float*>(output[pos_in_rhs]) [blockIdx.x] = RestOp<Float>()(smem[0]);
    }
}

}  //  nested namespace qcu::device::reduction
}
}
