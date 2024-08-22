#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include "public_complex_vector.h"
#include "qcu_macro.h"
#include <memory>
#include "qcu_blas/qcu_blas_elementwise_init.h"

using namespace std;

void elementwise_init_cpu ( Complex<InputFloat>* res,
                            Complex<InputFloat> scala,
                            int vec_len
                            )
{
  for (int i = 0; i < vec_len; ++i) {
    res[i] = scala;
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

  Complex<InputFloat>* cpu_res   = new Complex<InputFloat>[vector_length];
  Complex<InputFloat>* h_gpu_res = new Complex<InputFloat>[vector_length];

  Complex<InputFloat>*  gpu_res;


  CHECK_CUDA(cudaMalloc ((void**)&gpu_res, sizeof(Complex<InputFloat>) * vector_length));

  Complex<double> scala (rand() %1024, rand() % 1024);
  // GPU calculate
  using ElementwiseInitArg = qcu::qcu_blas::ElementwiseInit<Complex<InputFloat>>::ElementwiseInitArgument;

  ElementwiseInitArg arg (
    gpu_res,
    scala,
    vector_length,
    nullptr
  );

  qcu::qcu_blas::ElementwiseInit<Complex<InputFloat>> elementwise_init;
  elementwise_init(arg);
  CHECK_CUDA(cudaStreamSynchronize(nullptr));

  // cpu calculate
  // elementwise_div_cpu(cpu_res, cpu_x, cpu_y, vector_length); 
  elementwise_init_cpu(cpu_res, scala, vector_length);

  CHECK_CUDA(cudaMemcpy(h_gpu_res, gpu_res, sizeof(Complex<InputFloat>) * vector_length, cudaMemcpyDeviceToHost));
  cout << check_caxpby(cpu_res, h_gpu_res, vector_length) << endl;


  delete[] cpu_res;
  delete[] h_gpu_res;

  CHECK_CUDA(cudaFree(gpu_res));
}