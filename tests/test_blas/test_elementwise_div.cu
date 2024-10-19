#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include "public_complex_vector.h"
#include "qcu_public.h"
#include <memory>
#include "qcu_blas/qcu_blas_elementwise_div.h"
#include "check_error/check_cuda.cuh"

using namespace std;

void elementwise_div_cpu (Complex<InputFloat>* res,
                      Complex<InputFloat>* x,
                      Complex<InputFloat>* y,
                      int vec_len
                      )
{
  for (int i = 0; i < vec_len; ++i) {
    res[i] = x[i] / y[i];
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

  Complex<InputFloat>* cpu_x     = new Complex<InputFloat>[vector_length];
  Complex<InputFloat>* cpu_y     = new Complex<InputFloat>[vector_length];
  Complex<InputFloat>* cpu_res   = new Complex<InputFloat>[vector_length];
  Complex<InputFloat>* h_gpu_res = new Complex<InputFloat>[vector_length];

  Complex<InputFloat>*  gpu_x;
  Complex<InputFloat>*  gpu_y;
  Complex<InputFloat>*  gpu_res;

  CHECK_CUDA(cudaMalloc ((void**)&gpu_x,   sizeof(Complex<InputFloat>) * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&gpu_y,   sizeof(Complex<InputFloat>) * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&gpu_res, sizeof(Complex<InputFloat>) * vector_length));

  // init array 
  init_complex_randn_cpu(cpu_x, vector_length);
  init_complex_randn_cpu(cpu_y, vector_length);


  // COPY
  CHECK_CUDA(cudaMemcpy(gpu_x, cpu_x, sizeof(Complex<InputFloat>) * vector_length, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_y, cpu_y, sizeof(Complex<InputFloat>) * vector_length, cudaMemcpyHostToDevice));

  // GPU calculate
  using ElementwiseDivArg = qcu::qcu_blas::ElementwiseDiv<Complex<InputFloat>>::ElementwiseDivArgument;

  ElementwiseDivArg arg (
    gpu_res,
    gpu_x,
    gpu_y,
    vector_length,
    nullptr
  );

  qcu::qcu_blas::ElementwiseDiv<Complex<InputFloat>> elementwise_div;
  elementwise_div(arg);
  CHECK_CUDA(cudaStreamSynchronize(nullptr));

  // cpu calculate
  elementwise_div_cpu(cpu_res, cpu_x, cpu_y, vector_length); 

  CHECK_CUDA(cudaMemcpy(h_gpu_res, gpu_res, sizeof(Complex<InputFloat>) * vector_length, cudaMemcpyDeviceToHost));
  cout << check_caxpby(cpu_res, h_gpu_res, vector_length) << endl;

  delete[] cpu_x;
  delete[] cpu_y;
  delete[] cpu_res;
  delete[] h_gpu_res;
  CHECK_CUDA(cudaFree(gpu_x));
  CHECK_CUDA(cudaFree(gpu_y));
  CHECK_CUDA(cudaFree(gpu_res));
}