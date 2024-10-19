#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include "public_complex_vector.h"
#include "qcu_public.h"
#include <memory>
// #include "qcu_blas/qcu_blas.h"
#include "qcu_blas/qcu_blas_complex_xsay.h"
#include "check_error/check_cuda.cuh"

using namespace std;


void single_cxsay (Complex<InputFloat>* res,
                  Complex<InputFloat>* x, 
                  Complex<InputFloat>* a, 
                  Complex<InputFloat>* y, 
                  int single_vec_len, 
                  int inc_idx,
                  int start_idx
                  ) 
{
  for (int i = 0; i < single_vec_len; ++i) {
    res[i * inc_idx + start_idx] =  x[i * inc_idx + start_idx] 
                                  - a[start_idx] * y[i * inc_idx + start_idx];
  }
}

void array_cxsay (Complex<InputFloat>* res,
                  Complex<InputFloat>* x, 
                  Complex<InputFloat>* a, 
                  Complex<InputFloat>* y, 
                  int single_vec_len, 
                  int num_vecs
                  ) 
{
  for (int i = 0; i < num_vecs; ++i) {
    single_cxsay (res, x, a, y, single_vec_len, num_vecs, i);
  }
}

bool check_caxpby (Complex<InputFloat> * cpu_res, Complex<InputFloat> * gpu_res, int N) {
  for (int i = 0; i < N; ++i) {
    if (fabs((cpu_res[i] - gpu_res[i]).norm2()) > 1e-12) {
      cout << "elem pos: " << i << endl;
      cout << "ERROR ELEM: CPU(" << cpu_res[i].real() << "," << cpu_res[i].imag() << ")" << endl; 
      cout << "ERROR ELEM: GPU(" << gpu_res[i].real() << "," << gpu_res[i].imag() << ")" << endl; 
      return false;
    }
  }
  cout << "Correct" << endl;
  return true;
}

int main () {
  constexpr int num_vecs          = 3;
  const     int single_vec_length = 1024 * 1024 + 1;
  const     int vector_length     = single_vec_length * num_vecs;

  Complex<InputFloat>* cpu_a     = new Complex<InputFloat>[num_vecs];
  // Complex<InputFloat>* cpu_b     = new Complex<InputFloat>[num_vecs];
  Complex<InputFloat>* cpu_x     = new Complex<InputFloat>[vector_length];
  Complex<InputFloat>* cpu_y     = new Complex<InputFloat>[vector_length];
  Complex<InputFloat>* cpu_res   = new Complex<InputFloat>[vector_length];
  Complex<InputFloat>* h_gpu_res = new Complex<InputFloat>[vector_length];

  Complex<InputFloat>*  gpu_x;
  Complex<InputFloat>*  gpu_y;
  Complex<InputFloat>*  gpu_res;
  Complex<InputFloat>*  gpu_a;
  // Complex<InputFloat>*  gpu_b;

  CHECK_CUDA(cudaMalloc ((void**)&gpu_x,   sizeof(Complex<InputFloat>) * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&gpu_y,   sizeof(Complex<InputFloat>) * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&gpu_res, sizeof(Complex<InputFloat>) * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&gpu_a,   sizeof(Complex<InputFloat>) * num_vecs));
  // CHECK_CUDA(cudaMalloc ((void**)&gpu_b,   sizeof(Complex<InputFloat>) * num_vecs));

  // init array 
  // init_complex_range_1_N_cpu(cpu_a, num_vecs);
  // init_complex_range_1_N_cpu(cpu_b, num_vecs);
  // init_complex_1_2_cpu(cpu_x, vector_length);
  // init_complex_1_2_cpu(cpu_y, vector_length);
  init_complex_randn_cpu(cpu_a, num_vecs);
  // init_complex_randn_cpu(cpu_b, num_vecs);
  init_complex_randn_cpu(cpu_x, vector_length);
  init_complex_randn_cpu(cpu_y, vector_length);


  // COPY
  CHECK_CUDA(cudaMemcpy(gpu_x, cpu_x, sizeof(Complex<InputFloat>) * vector_length, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_y, cpu_y, sizeof(Complex<InputFloat>) * vector_length, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_a, cpu_a, sizeof(Complex<InputFloat>) * num_vecs,      cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(gpu_b, cpu_b, sizeof(Complex<InputFloat>) * num_vecs,      cudaMemcpyHostToDevice));
  // GPU calculate
  using CxsayArgument = qcu::qcu_blas::Complex_xsay<InputFloat>::Complex_xsayArgument;

  CxsayArgument arg (
    gpu_res,
    gpu_x,
    gpu_a,
    gpu_y,
    single_vec_length,
    num_vecs,
    nullptr
  );

  qcu::qcu_blas::Complex_xsay<InputFloat> cxsay;
  cxsay(arg);
  CHECK_CUDA(cudaStreamSynchronize(nullptr));

  // cpu calculate
  array_cxsay(cpu_res, cpu_x, cpu_a, cpu_y, single_vec_length, num_vecs); 

  CHECK_CUDA(cudaMemcpy(h_gpu_res, gpu_res, sizeof(Complex<InputFloat>) * vector_length, cudaMemcpyDeviceToHost));
  cout << check_caxpby(cpu_res, h_gpu_res, vector_length) << endl;

  delete[] cpu_a;
  // delete[] cpu_b;
  delete[] cpu_x;
  delete[] cpu_y;
  delete[] cpu_res;
  delete[] h_gpu_res;
  CHECK_CUDA(cudaFree(gpu_x));
  CHECK_CUDA(cudaFree(gpu_y));
  CHECK_CUDA(cudaFree(gpu_res));
  CHECK_CUDA(cudaFree(gpu_a));
  // CHECK_CUDA(cudaFree(gpu_b));
}