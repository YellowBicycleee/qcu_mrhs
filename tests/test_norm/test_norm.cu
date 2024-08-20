#include <iostream>
#include <vector>
#include <cublas_v2.h>
// #include "complex/qcu_complex.cuh"
#include "qcu_blas_public.h"
#include "qcu_blas/qcu_blas.h"
#include <type_traits>
#include "qcu_macro.h"
#include "timer/timer.h"
// #include <complex>

using namespace std;

using InputFloat  = double;
using OutputFloat = double;

bool check_correct (const vector<OutputFloat> & a, const vector<OutputFloat> & b) {
  int size = a.size();
  if (a.size() != b.size()) {
    return false;
  }
  // bool res = true;
  for (int i = 0; i < size; ++i) {
    if (fabs(a[i] - b[i]) > 1e-12) 
    {
      cout << "Error" << endl;;
      return false;
    }
  }
  std::cout << "Correct" << std::endl;
  // for (auto elem : a) {
  //   cout << elem << " ";
  // }
  cout << "===============" << endl;
  return true;
}

void init_complex (Complex<double>* a, int N) 
{
  for (int i = 0; i < N; ++i ) {
    a[i] = Complex<double> (rand()%16, rand()%16 + 1);
    // a[i] = Complex<double> (1, 0);
  }
}

OutputFloat cpu_norm (Complex<InputFloat>* h_in, 
                      int single_vec_length, 
                      int stride) 
{
  OutputFloat rank_res = 0;
  for (int i = 0; i < single_vec_length; ++i) {
    rank_res += (h_in[i * stride].norm2());
  }
  return rank_res;
}

void cpu_norm_vec ( vector<OutputFloat> & res,
                    Complex<InputFloat> * h_in, 
                    int single_vec_length, 
                    int stride) 
{
  res.clear();
  // OutputFloat rank_res = 0;
  for (int i = 0; i < stride; ++i) {
    res.push_back(sqrt(cpu_norm(h_in + i, single_vec_length, stride)));
  }
}

void cublas_single_vec_norm ( OutputFloat*         d_res, 
                              Complex<InputFloat>* d_input,
                              int                  single_vec_length,
                              int                  stride,
                              cublasHandle_t       cublas_handle
                              )
{
  if constexpr (std::is_same_v<InputFloat, double>) 
  {
    QCU_CHECK_CUBLAS (
          cublasDznrm2( cublas_handle,
                        single_vec_length,
                        reinterpret_cast<cuDoubleComplex*>(d_input), 
                        stride,
                        reinterpret_cast<double*>(d_res))
    );
    
  } 
  else if constexpr (std::is_same_v<InputFloat, float>) 
  {
    QCU_CHECK_CUBLAS (
          cublasScnrm2( cublas_handle,
                        single_vec_length,
                        reinterpret_cast<cuComplex*>(d_input), 
                        stride,
                        reinterpret_cast<float*>(d_res))
    );
  }
} 

void cublas_arr_vec_norm (vector<OutputFloat> & h_gpu_res,
                          OutputFloat*          d_res, 
                          Complex<InputFloat>*  d_input,
                          int                   single_vec_length,
                          int                   stride,
                          cublasHandle_t        cublas_handle
                          )
{
  OutputFloat        *  d_out;
  Complex<InputFloat>*  d_in;
  h_gpu_res.clear();
  for (int i = 0; i < stride; ++i) {
    d_out = d_res   + i;
    d_in  = d_input + i;
    cublas_single_vec_norm(d_out, d_in, single_vec_length, stride, cublas_handle);
  }
  h_gpu_res.resize(stride);
  CHECK_CUDA( cudaMemcpy( h_gpu_res.data(), 
                          d_res, 
                          sizeof(OutputFloat) * stride,
                          cudaMemcpyDeviceToHost));
}

void my_function (vector<OutputFloat> & h_gpu_res,
                  OutputFloat*          d_res, 
                  OutputFloat*          d_temp_buffer,
                  Complex<InputFloat>*  d_input,
                  int                   single_vec_length,
                  int                   stride,
                  cudaStream_t          stream,
                  cublasHandle_t        cublas_handle
                  )
{
  // OutputFloat        *  d_out;
  // Complex<InputFloat>*  d_in;
  h_gpu_res.clear();
  h_gpu_res.resize(stride);
  
  using Argument = qcu::qcu_blas::ComplexNorm<OutputFloat, InputFloat>::ComplexNormArgument;
  Argument arg {
    single_vec_length,
    stride,
    d_temp_buffer,
    d_input,
    d_res,
    stream,
    cublas_handle
  };
  qcu::qcu_blas::ComplexNorm<OutputFloat, InputFloat> norm;

  norm(arg);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA( cudaMemcpy( h_gpu_res.data(), 
                          d_res, 
                          sizeof(OutputFloat) * stride,
                          cudaMemcpyDeviceToHost));

}


int main () {

  constexpr int num_vecs          = 12;
  const     int single_vec_length = 1024 * 1024;
  const     int vector_length     = single_vec_length * num_vecs;

  Complex<InputFloat>*  cpu_in    = new Complex<InputFloat>[vector_length];
  vector<OutputFloat>   h_cpu_res_vec(num_vecs, 0);

  Complex<InputFloat>*  gpu_in;
  vector<OutputFloat>   h_gpu_res_vec(num_vecs, 0);
  OutputFloat*          gpu_out;
  OutputFloat*          gpu_temp_buffer;

  Complex<InputFloat>*  my_gpu_in;
  vector<OutputFloat>   my_h_gpu_res_vec(num_vecs, 0);
  OutputFloat*          my_gpu_out;
  OutputFloat*          my_gpu_temp_buffer;
  // cudaMalloc
  CHECK_CUDA(cudaMalloc ((void**)&gpu_in,          sizeof(Complex<InputFloat>) * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&gpu_temp_buffer, sizeof(OutputFloat)         * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&gpu_out,         sizeof(OutputFloat)         * vector_length));

  CHECK_CUDA(cudaMalloc ((void**)&my_gpu_in,          sizeof(Complex<InputFloat>) * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&my_gpu_temp_buffer, sizeof(OutputFloat)         * vector_length));
  CHECK_CUDA(cudaMalloc ((void**)&my_gpu_out,         sizeof(OutputFloat)         * vector_length));

  // init arr
  init_complex(cpu_in, vector_length);
  // copy
  CHECK_CUDA(cudaMemcpy(gpu_in, cpu_in, sizeof(Complex<OutputFloat>) * vector_length, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(my_gpu_in, cpu_in, sizeof(Complex<OutputFloat>) * vector_length, cudaMemcpyHostToDevice));
  // cublas init
  cublasHandle_t   cublas_handle;
  QCU_CHECK_CUBLAS (cublasCreate(&cublas_handle));
  QCU_CHECK_CUBLAS (cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

  // warm up
  cublas_arr_vec_norm(h_gpu_res_vec, 
                      gpu_out, 
                      gpu_in, 
                      single_vec_length, 
                      num_vecs, 
                      cublas_handle);
  
  Timer timer;
  timer.start();
  // cublas calcualtion
  cublas_arr_vec_norm(h_gpu_res_vec, 
                      gpu_out, 
                      gpu_in, 
                      single_vec_length, 
                      num_vecs, 
                      cublas_handle);
  timer.stop();
  cout << "cublas time      = " << timer.getElapsedTimeSecond() << endl;

  timer.start();
  // cpu
  cpu_norm_vec(h_cpu_res_vec, cpu_in, single_vec_length, num_vecs);
  timer.stop();
  cout << "cpu time         = " << timer.getElapsedTimeSecond() << endl;

  timer.start();
  // my_function
  cudaStream_t stream = nullptr;
  my_function(my_h_gpu_res_vec, 
              my_gpu_out, 
              my_gpu_temp_buffer, 
              my_gpu_in, 
              single_vec_length, 
              num_vecs, 
              stream, 
              cublas_handle);
  timer.stop();
  cout << "my_function time = " << timer.getElapsedTimeSecond() << endl;

  // check
  // cout << "h_cpu_size = " << h_cpu_res_vec.size() << "h_gpu_size = " << h_gpu_res_vec.size() << endl;
  cout << check_correct(h_cpu_res_vec, h_gpu_res_vec) << endl;
  cout << check_correct(h_gpu_res_vec, my_h_gpu_res_vec) << endl;
  // for (int i = 0; i < h_cpu_res_vec.size(); ++i) {
  //   cout << h_cpu_res_vec[i] << " " << h_gpu_res_vec[i] << endl;
  // }
  for (int i = 0; i < h_cpu_res_vec.size(); ++i) {
    cout  << "cpu   : " <<  h_cpu_res_vec[i] << " "
          << "cublas: " << h_gpu_res_vec[i] << " "
          << "my GPU: " << my_h_gpu_res_vec[i] << endl;
  }
  cudaFree(gpu_in);
  cudaFree(gpu_temp_buffer);
  cudaFree(gpu_out);
  
  QCU_CHECK_CUBLAS (cublasDestroy(cublas_handle));
  delete[] cpu_in;

}