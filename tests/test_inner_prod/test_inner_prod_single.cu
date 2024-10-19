#include <iostream> 
#include <complex>
#include <cublas_v2.h>
#include "timer.h"
#include <vector>
#include <numeric>
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

complex<float> cpu_inner_dot_c (complex<float>* a, complex<float>* b, int N, int stride) {
  complex<float> res = 0;
  for (int i = 0; i < N; ++i) {
    res += (conj(a[i * stride]) * b[i * stride]);
  }
  return res;
}

void init_complex_1 (complex<float>* a, int N) {
  for (int i = 0; i < N; ++i ) {
    a[i] = complex<float> (rand()%16, rand()%16 + 1);
  }
}
void init_complex_2 (complex<float>* a, int N) {
  for (int i = 0; i < N; ++i ) {
    a[i] = complex<float> (rand()%16, rand()%16 + 1);
  }
}

bool check_correct (const vector<complex<float>>& a, const vector<complex<float>>& b) {
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
  complex<float>* h_a;
  complex<float>* h_b;
  complex<float> h_res;
  complex<float> d_res;

  complex<float>* d_a;
  complex<float>* d_b;
  complex<float>* dd_res;

  // malloc 
  const int single_vec_length = 16 * 16 * 16 * 32 * 12;
  const int num_vecs = 8;
  const int vector_length = single_vec_length * num_vecs;

  cout << "vec size = " << single_vec_length * sizeof(complex<float>) / 1024 / 1024 << "MB" << endl;
  vector<complex<float>> d_res_vec (num_vecs);
  vector<complex<float>> h_res_vec (num_vecs);


  h_a = new complex<float>[vector_length];
  h_b = new complex<float>[vector_length];

  cudaMalloc ((void**)&d_a, sizeof(complex<float>) * vector_length);
  cudaMalloc ((void**)&d_b, sizeof(complex<float>) * vector_length);
  cudaMalloc ((void**)&dd_res, sizeof(complex<float>));

  // init arr
  init_complex_1(h_a, vector_length);
  init_complex_2(h_b, vector_length);
  // copy
  cudaMemcpy(d_a, h_a, sizeof(complex<float>) * vector_length, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(complex<float>) * vector_length, cudaMemcpyHostToDevice);




  // gpu
  // cublasStatus_t cublas_stat;
  cublasHandle_t cublas_handle;

  CHECK_CUBLAS (cublasCreate(&cublas_handle));

    // warm up
  CHECK_CUBLAS (cublasCdotc(cublas_handle, vector_length, 
                        reinterpret_cast<const cuFloatComplex*>(d_a), 1, 
                        reinterpret_cast<const cuFloatComplex*>(d_b), 1, 
                        reinterpret_cast<cuFloatComplex*>(dd_res)
                  ));


  TIMER_EVENT (
    CHECK_CUBLAS (cublasCdotc(cublas_handle, vector_length, 
                        reinterpret_cast<const cuFloatComplex*>(d_a), 1, 
                        reinterpret_cast<const cuFloatComplex*>(d_b), 1, 
                        reinterpret_cast<cuFloatComplex*>(dd_res)
                  )
    ), 
    "cublas                     : "
  );
  // cublasDestroy(cublas_handle);
  cudaMemcpy(&d_res, dd_res, sizeof(complex<float>), cudaMemcpyDeviceToHost);
  // cpu
  TIMER_EVENT (
    h_res = cpu_inner_dot_c(h_a, h_b, vector_length, 1)
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
      CHECK_CUBLAS (cublasCdotc(cublas_handle, single_vec_length, 
                      reinterpret_cast<const cuFloatComplex*>(d_a + i), num_vecs, 
                      reinterpret_cast<const cuFloatComplex*>(d_b + i), num_vecs, 
                      reinterpret_cast<cuFloatComplex*>(dd_res)
                )
      );
      cudaMemcpy(&d_res, dd_res, sizeof(complex<float>), cudaMemcpyDeviceToHost);
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
  cout << "stride = " << num_vecs << ", res = " << std::accumulate(d_res_vec.begin(), d_res_vec.end(), complex<float>(0, 0)) << endl;
  cout << "stride = " << num_vecs << ", res = " << std::accumulate(h_res_vec.begin(), h_res_vec.end(), complex<float>(0, 0)) << endl;

  // stride = 1 res
  cout << "=========stride inner pro, stride = 1 =============" << endl;
  d_res_vec.clear();
  h_res_vec.clear();
  // gpu stride innerproduct
  TIMER_EVENT (
    for (int i = 0; i < num_vecs; ++i) {
      CHECK_CUBLAS (cublasCdotc(cublas_handle, single_vec_length, 
                      reinterpret_cast<const cuFloatComplex*>(d_a + i * single_vec_length), 1, 
                      reinterpret_cast<const cuFloatComplex*>(d_b + i * single_vec_length), 1, 
                      reinterpret_cast<cuFloatComplex*>(dd_res)
                )
      );
      // cudaDeviceSynchronize();
      cudaMemcpy(&d_res, dd_res, sizeof(complex<float>), cudaMemcpyDeviceToHost);
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
        // h_res_vec.push_back(h_res);
        h_res_vec.push_back(h_res);
        // cout << h_res_vec.size() << endl;
    }
    , 
    "cpu, stride innerproduct        : "
  );

  cout << "if_res_correct? : " << check_correct(d_res_vec, h_res_vec) << endl;
  cout << "stride = 1, res = " << std::accumulate(d_res_vec.begin(), d_res_vec.end(), complex<float>(0, 0)) << endl;
  cout << "stride = 1, res = " << std::accumulate(h_res_vec.begin(), h_res_vec.end(), complex<float>(0, 0)) << endl;

  cout << check_correct(d_res_vec, vector<complex<float>>{});
  CHECK_CUBLAS (cublasDestroy(cublas_handle));
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(dd_res);
  delete[] h_a;
  delete[] h_b;
}
