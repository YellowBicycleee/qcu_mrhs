#pragma once

// shift 1D index to 4D index
inline __device__ __host__ void get4DCoord(int &t, int &z, int &y, int &x, int lex_id, int Lz, int Ly,
                                                           int Lx) {
    t = lex_id / (Lz * Ly * Lx);
    z = lex_id % (Lz * Ly * Lx) / (Ly * Lx);
    y = lex_id % (Ly * Lx) / Lx;
    x = lex_id % Lx;
}

inline __device__ __host__ int index_2D(int y, int x, int Lx) { return y * Lx + x; }
inline __device__ __host__ int index_3D(int z, int y, int x, int Ly, int Lx) {
    return (z * Ly + y) * Lx + x;
}
inline __device__ __host__ int index_4D(int t, int z, int y, int x, int Lz, int Ly, int Lx) {
    return ((t * Lz + z) * Ly + y) * Lx + x;
}

inline __device__ __host__ int div_ceil(int a, int b) { return (a + b - 1) / b; }

int get_gpu_id();
int get_process_rank();