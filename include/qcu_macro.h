#pragma once
#include <cstdio>

// #define PROFILE_DEBUG


constexpr int MAX_DIM = 4;
constexpr int WARP_SIZE = 32;
constexpr int WARP_PER_BLOCK = 1;

#define IDX2D(y, x, lx) ((y) * (lx) + (x))
#define IDX3D(z, y, x, ly, lx) ((((z) * (ly)) + (y)) * (lx) + (x))
#define IDX4D(t, z, y, x, lz, ly, lx) ((((t) * (lz) + (z)) * (ly) + (y)) * (lx) + (x))

#define NOT_IMPLEMENTED "Not implemented yet\n" 

inline int mul(int x, int y, int z = 1, int t = 1) { return x * y * z * t; }

constexpr int Ns = 4;

#define CHECK_MPI(cmd)                               \
    do {                                             \
        int err = cmd;                               \
        if (err != MPI_SUCCESS) {                    \
            fprintf(stderr, "MPI error: %d\n", err); \
            exit(1);                                 \
        }                                            \
    } while (0)



#ifdef PROFILE_DEBUG

#define CHECK_CUDA(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while (0) 

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define CHECK_CUDA(ans) ans
#endif


// #define CHECK_CUDA(cmd)                                                                                         \
//     do {                                                                                                        \
//         cudaError_t err = cmd;                                                                                  \
//         if (err != cudaSuccess) {                                                                               \
//             fprintf(stderr, "CUDA error: %s, file %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
//             exit(1);                                                                                            \
//         }                                                                                                       \
//     } while (0)

#define CHECK_NCCL(cmd)                                                   \
    do {                                                                  \
        ncclResult_t err = cmd;                                           \
        if (err != ncclSuccess) {                                         \
            fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(err)); \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define errorQcu(msg)                                                                \
    do {                                                                             \
        fprintf(stderr, msg);                                                        \
        fprintf(stderr, "Error happened in file %s, line %d\n", __FILE__, __LINE__); \
        exit(1);                                                                     \
    } while (0)
