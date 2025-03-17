#pragma once

namespace qcu::arch {

__device__ __forceinline__ int lane_id () {
    int id;
    asm("mov.u32 %0, %laneid;" : "=r"(id));
    return id;
}

__device__ __forceinline__ int warp_id () {
    int id;
    asm("mov.u32 %0, %warpid;" : "=r"(id));
    return id;
}

}