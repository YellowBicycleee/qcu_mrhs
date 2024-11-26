#pragma once

__forceinline__ __device__ void cuda_abort() {
    asm("trap;"); // Abort execution and generate an interrupt to the host CPU
}
