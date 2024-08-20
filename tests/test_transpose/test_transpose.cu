#include <iostream> 
#include <complex>
#include "timer.h"
#include <vector>
#include <numeric>
#include "qcu_blas/qcu_transpose_2d.h"
using namespace std;



void cpu_transpose (complex<double>* out, complex<double>* in, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      out[j * m + i] = in[i * n + j];
    }
  }
}

void init_complex_1 (complex<double>* a, int N) {
  for (int i = 0; i < N; ++i ) {
    a[i] = complex<double> (rand()%16, rand()%16 + 1);
  }
}

template <typename T>
bool check_correct (const T* a, const T* b, int m, int n) {
  int res = true;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (fabs(a[i * n + j].real() - b[i * n + j].real()) > 1e-12 || 
          fabs(a[i * n + j].imag() - b[i * n + j].imag()) > 1e-12 
      ) {
        // cout << "i = " << i << " ,j = " << j << endl;
        // cout << "a[i * n + j] = " << a[i * n + j] << endl;
        // cout << "b[i * n + j] = " << b[i * n + j] << endl;
        res = false;
      }
    }
  }
  return res;
}

int main () {
  complex<double>* h_a;
  complex<double>* h_b;
  complex<double>* h_d_b;

  complex<double>* d_a;
  complex<double>* d_b;

  // malloc 
  const int single_vec_length = 1024 * 1024;
  const int num_vecs = 20;
  const int vector_length = single_vec_length * num_vecs;

  vector<complex<double>> d_res_vec (num_vecs);
  vector<complex<double>> h_res_vec (num_vecs);

  h_a = new complex<double>[vector_length];
  h_b = new complex<double>[vector_length];
  h_d_b = new complex<double>[vector_length];

  cudaMalloc ((void**)&d_a, sizeof(complex<double>) * vector_length);
  cudaMalloc ((void**)&d_b, sizeof(complex<double>) * vector_length);

  // init arr
  init_complex_1(h_a, vector_length);
  // copy
  cudaMemcpy(d_a, h_a, sizeof(complex<double>) * vector_length, cudaMemcpyHostToDevice);
  // gpu_transpose
  // using namespace qcu::qcu_blas;
  
  
  qcu::qcu_blas::Transpose2D<double2> trans2d (
    qcu::qcu_blas::Transpose2D<double2>::Transpose2DParam
        { single_vec_length, 
          num_vecs, 
          static_cast<void*>(d_b), 
          static_cast<void*>(d_a) 
        }
  );
  TIMER_EVENT (
    {
      trans2d ();
      cudaDeviceSynchronize();
    }, 
    "T, 2d"
  );
  // cpu
  cpu_transpose(h_b, h_a, single_vec_length, num_vecs);

  cudaMemcpy (h_d_b, d_b, sizeof(complex<double>) * vector_length, cudaMemcpyDeviceToHost);
  cout << "check correct: " << check_correct(h_b, h_d_b, single_vec_length, num_vecs) << endl;

  // for (int i = 0; i < 5; ++i) {
  //   for (int j = 0; j < 5; ++j) {
  //     cout << h_b[i * num_vecs + j] << " ";
  //   }
  //   cout << endl;
  // }
  // cout << "======================" << endl;
  // for (int i = 0; i < 5; ++i) {
  //   for (int j = 0; j < 5; ++j) {
  //     cout << h_d_b[i * num_vecs + j] << " ";
  //   }
  //   cout << endl;
  // }
  // cout << "======================" << endl;
  // for (int i = 0; i < 5; ++i) {
  //   for (int j = 0; j < 5; ++j) {
  //     cout << h_a[i * num_vecs + j] << " ";
  //   }
  //   cout << endl;
  // }


  // cout << h_b[10] << " " << h_d_b[10] << endl;  

  // for (int i = 0; i < vector_length; ++i) {
  //   cout << h_b[i] << " " << h_d_b[i] << endl;
  // }
  cudaFree(d_a);
  cudaFree(d_b);
  delete[] h_a;
  delete[] h_b;
}