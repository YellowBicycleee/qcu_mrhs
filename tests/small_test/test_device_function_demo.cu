#include <cstdio>
#include <chrono>

__device__ void Add (float &c, const float &a, const float &b) {
    c = a + b;
}
__device__ void Add1 (float &c, const float &a, const float &b) {
    Add(c, a, b);
}

__device__ void Add2 (float &c, const float &a, const float &b) {
    Add1(c, a, b);
}

__global__ void vectorAddFunc (float *a, float * __restrict__ b, float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        Add2(c[i], a[i], b[i]);
    }
}

__global__ void vectorAddNaive (float *a, float * __restrict__ b, float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}


int main () {
    size_t size = 1 << 20;

    float *a, *b, *c;
    cudaMalloc(&a, size * sizeof(float));
    cudaMalloc(&b, size * sizeof(float));
    cudaMalloc(&c, size * sizeof(float));
    

    auto start = std::chrono::high_resolution_clock::now();  
    vectorAddNaive<<<1, 1024>>>(a, b, c, size);
    cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); 
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); 
	printf("naive takes %e seconds, ': \n", double(duration) / 1e9);
	printf("GFLOPS = %e", size / (double(duration) / 1e9) / 1e9);

    printf("===================separator===================\n");

    start = std::chrono::high_resolution_clock::now();
    vectorAddFunc<<<1, 1024>>>(a, b, c, size);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("func takes %e seconds, ': \n", double(duration) / 1e9);
    printf("GFLOPS = %e", size / (double(duration) / 1e9) / 1e9);
    
}