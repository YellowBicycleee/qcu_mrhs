#include <iostream> 
#include <complex>
#include <cublas_v2.h>
#include "timer.h"
#include <vector>
#include <numeric>

#include "qcu_blas/qcu_blas.h"
#include "complex/qcu_complex.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

#define CHECK_CUBLAS(cmd)   \
do {  \
  cublasStatus_t stat = cmd;      \
  check_cublas (stat, __FILE__, __LINE__);\
} while (0)

inline void check_cublas (cublasStatus_t stat, const char* file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("IN file %s, line %d, error happened\n", file, line);
    abort();
  }
}

complex<double> cpu_inner_dot_c (complex<double>* a, complex<double>* b, int N, int stride) {
  complex<double> res = 0;
  for (int i = 0; i < N; ++i) {
    res += (conj(a[i * stride]) * b[i * stride]);
  }
  return res;
}

// void init_complex_1 (complex<double>* a, int N) {
//   for (int i = 0; i < N; ++i ) {
//     // a[i] = complex<double> (2 * i % 32, 2 * i % 32 + 1);
//     a[i] = complex<double> (rand()%16, rand()%16 + 1);
//   }
// }
void init_complex (complex<double>* a, int N) {
  for (int i = 0; i < N; ++i ) {
    // a[i] = complex<double> (2 * i % 16, 2 * i % 16 + 1);
    a[i] = complex<double> (rand()%16, rand()%16 + 1);
    // a[i] = complex<double> (1, 0);
  }
}

bool check_correct (const vector<complex<double>>& a, const vector<complex<double>>& b) {
  int size = a.size();
  if (a.size() != b.size()) {
    return false;
  }
  // bool res = true;
  for (int i = 0; i < size; ++i) {
    if (fabs(a[i].real() - b[i].real()) > 1e-12 || 
        fabs(a[i].imag() - b[i].imag()) > 1e-12 
    ) {
      return false;
    }
  }
  return true;
}

int main () {

  complex<double>* h_a;
  complex<double>* h_b;
  complex<double> h_res;
  complex<double> d_res;

  complex<double>* d_a;
  complex<double>* d_b;
  complex<double>* dd_res_arr;
  complex<double>* d_temp;



  // malloc 
  const int single_vec_length = 1024 * 1024;
  const int num_vecs = 12;
  const int vector_length = single_vec_length * num_vecs;
  

  vector<complex<double>> d_res_vec (num_vecs);
  vector<complex<double>> h_res_vec (num_vecs);


  h_a = new complex<double>[vector_length];
  h_b = new complex<double>[vector_length];

  cudaMalloc ((void**)&d_a, sizeof(complex<double>) * vector_length);
  cudaMalloc ((void**)&d_b, sizeof(complex<double>) * vector_length);
  cudaMalloc ((void**)&d_temp, sizeof(complex<double>) * vector_length);
  cudaMalloc ((void**)&dd_res_arr, sizeof(complex<double>) * num_vecs);

  // init arr
  init_complex(h_a, vector_length);
  init_complex(h_b, vector_length);
  // copy
  cudaMemcpy(d_a, h_a, sizeof(complex<double>) * vector_length, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(complex<double>) * vector_length, cudaMemcpyHostToDevice);




  // gpu
  // cublasStatus_t cublas_stat;
  cublasHandle_t cublas_handle;

  CHECK_CUBLAS (cublasCreate(&cublas_handle));

    // warm up
  CHECK_CUBLAS (cublasZdotc(cublas_handle, vector_length, 
                        reinterpret_cast<const cuDoubleComplex*>(d_a), 1, 
                        reinterpret_cast<const cuDoubleComplex*>(d_b), 1, 
                        reinterpret_cast<cuDoubleComplex*>(dd_res_arr)
                  ));


  TIMER_EVENT (
    {
      CHECK_CUBLAS (cublasZdotc(cublas_handle, vector_length, 
                          reinterpret_cast<const cuDoubleComplex*>(d_a), 1, 
                          reinterpret_cast<const cuDoubleComplex*>(d_b), 1, 
                          reinterpret_cast<cuDoubleComplex*>(dd_res_arr)
                    )
      ); 
      cudaDeviceSynchronize();
    }, 
    "cublas                     : "
  );

  using DotcArgument = qcu::qcu_blas::ComplexDotc<double, double>::DotcArgument;
  DotcArgument arg {
        vector_length,
        1,
        reinterpret_cast<Complex<double>*>(d_temp),
        reinterpret_cast<Complex<double>*>(d_a),
        reinterpret_cast<Complex<double>*>(d_b),
        reinterpret_cast<Complex<double>*>(dd_res_arr),
        nullptr,
        cublas_handle
  };

  qcu::qcu_blas::ComplexDotc<double, double> dotc;

  TIMER_EVENT(
    {  
      dotc(arg);
      cudaDeviceSynchronize();
      cudaMemcpy(&d_res, dd_res_arr, sizeof(complex<double>), cudaMemcpyDeviceToHost);
    }
    , "my dotc                    : "
  );
  // cpu
  TIMER_EVENT (
    h_res = cpu_inner_dot_c(h_a, h_b, vector_length/1, 1)
    , 
    "cpu                        : "
  );

  std::cout << "cpu_res: " << h_res << std::endl;
  std::cout << "gpu_res: " << d_res << std::endl;

  d_res_vec.clear();
  h_res_vec.clear();
  cout << "=========stride inner pro, stride = num_vecs =============" << endl;
  // gpu stride innerproduct
  TIMER_EVENT (
    for (int i = 0; i < num_vecs; ++i) {
      CHECK_CUBLAS (cublasZdotc(cublas_handle, single_vec_length, 
                      reinterpret_cast<const cuDoubleComplex*>(d_a + i), num_vecs, 
                      reinterpret_cast<const cuDoubleComplex*>(d_b + i), num_vecs, 
                      reinterpret_cast<cuDoubleComplex*>(dd_res_arr)
                )
      );
      cudaMemcpy(&d_res, dd_res_arr, sizeof(complex<double>), cudaMemcpyDeviceToHost);
      d_res_vec.push_back(d_res);
    }
    , 
    "cubulas, stride innerproduct    : "
  );

  // stride innerproduct
  TIMER_EVENT (
    for (int i = 0; i < num_vecs; ++i) {
        h_res = cpu_inner_dot_c(h_a + i, h_b + i, single_vec_length, num_vecs);
        h_res_vec.push_back(h_res);
    }
    ,
    "cpu, stride innerproduct        : "
  );
  cout << "if_res_correct? : " << check_correct(d_res_vec, h_res_vec) << endl;
  cout << "stride = " << num_vecs << ", res = " << std::accumulate(d_res_vec.begin(), d_res_vec.end(), complex<double>(0, 0)) << endl;
  cout << "stride = " << num_vecs << ", res = " << std::accumulate(h_res_vec.begin(), h_res_vec.end(), complex<double>(0, 0)) << endl;

  // stride = 1 res
  cout << "=========stride inner pro, stride = num_vecs =============" << endl;
  d_res_vec.clear();
  h_res_vec.clear();
  // gpu stride innerproduct
  TIMER_EVENT (
    for (int i = 0; i < num_vecs; ++i) {
      CHECK_CUBLAS (cublasZdotc(cublas_handle, single_vec_length, 
                      reinterpret_cast<const cuDoubleComplex*>(d_a + i * single_vec_length), 1, 
                      reinterpret_cast<const cuDoubleComplex*>(d_b + i * single_vec_length), 1, 
                      reinterpret_cast<cuDoubleComplex*>(dd_res_arr)
                )
      );
      // cudaDeviceSynchronize();
      cudaMemcpy(&d_res, dd_res_arr, sizeof(complex<double>), cudaMemcpyDeviceToHost);
      d_res_vec.push_back(d_res);
    }
    , 
    "cubulas, stride = 1 innerproduct: "
  );

  // cpu 
    // stride innerproduct
  TIMER_EVENT (
    for (int i = 0; i < num_vecs; ++i) {
        h_res = cpu_inner_dot_c(h_a + i * single_vec_length, h_b + i * single_vec_length, single_vec_length, 1);
        h_res_vec.push_back(h_res);
    }
    , 
    "cpu, stride innerproduct        : "
  );

  cout << "if_res_correct? : " << check_correct(d_res_vec, h_res_vec) << endl;
  cout << "stride = 1, res = " << std::accumulate(d_res_vec.begin(), d_res_vec.end(), complex<double>(0, 0)) << endl;
  cout << "stride = 1, res = " << std::accumulate(h_res_vec.begin(), h_res_vec.end(), complex<double>(0, 0)) << endl;

  cout << check_correct(d_res_vec, vector<complex<double>>{});
  CHECK_CUBLAS (cublasDestroy(cublas_handle));
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_temp);
  cudaFree(dd_res_arr);
  delete[] h_a;
  delete[] h_b;
}
