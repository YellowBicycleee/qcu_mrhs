#pragma once

#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define QCU_ARCH_WMMA_ENABLED
#define QCU_ARCH_WMMA_SM80_ENABLED
#endif


__forceinline__ __device__ void cuda_abort() {
    asm("trap;"); // Abort execution and generate an interrupt to the host CPU
}
