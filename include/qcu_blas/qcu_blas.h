#pragma once
#include "complex/qcu_complex.cuh"
#include <cublas_v2.h>

namespace qcu::qcu_blas {

template <typename OutputFloat, typename InputFloat>
struct ComplexNorm {
  // Argument type
  struct ComplexNormArgument {
    int                     single_vec_len;
    int                     stride;
    OutputFloat         *   tmpBuffer;  // reduce结果一般都很短，tmpBuffer作为第一次的输出，第二次的输入+输出
                                        // when you use cublas, you can set it to nullptr
    Complex<InputFloat> *   input;
    OutputFloat         *   resArr;  // [device_res1, device_res2, ...]
    cudaStream_t            stream;
    cublasHandle_t          handle;

    ComplexNormArgument(
      int                     single_vec_len,
      int                     stride    = 1,
      OutputFloat         *   tmpBuffer = nullptr,
      Complex<InputFloat> *   input     = nullptr,
      OutputFloat         *   resArr    = nullptr,
      cudaStream_t            stream    = nullptr,
      cublasHandle_t          handle    = nullptr
    ) : single_vec_len(single_vec_len),
        stride(stride),
        tmpBuffer(tmpBuffer),
        input(input),
        resArr(resArr),
        stream(stream),
        handle(handle) {} 
  };
  void operator () (ComplexNormArgument);
};

template <typename OutputFloat, typename InputFloat>
struct ComplexDotc {
  // Argument type 
  struct DotcArgument {
    int                     single_vec_len;
    int                     stride;
    Complex<OutputFloat>*   tmpBuffer;  // reduce结果一般都很短，tmpBuffer作为第一次的输出，第二次的输入+输出
                                        // when you use cublas, you can set it to nullptr
    Complex<InputFloat> *   input1;
    Complex<InputFloat> *   input2;
    Complex<OutputFloat>*   resArr;  // [device_res1, device_res2, ...]
    cudaStream_t            stream;
    cublasHandle_t          handle;

    DotcArgument(
      int                     single_vec_len,
      int                     stride    = 1,
      Complex<OutputFloat>*   tmpBuffer = nullptr,
      Complex<InputFloat> *   input1    = nullptr,
      Complex<InputFloat> *   input2    = nullptr,
      Complex<OutputFloat>*   resArr    = nullptr,
      cudaStream_t            stream    = nullptr,
      cublasHandle_t          handle    = nullptr
    ) : single_vec_len(single_vec_len),
        stride(stride),
        tmpBuffer(tmpBuffer),
        input1(input1),
        input2(input2),
        resArr(resArr),
        stream(stream),
        handle(handle) {} 
  };

  // methods
  void operator () (DotcArgument);
};


template <typename _Tp>
struct Complex_axpby {
  // Argument type
  struct ComplexAxpbyArgument {
    int                     single_vec_len;
    int                     stride;
    Complex<_Tp>            a;
    Complex<_Tp>            b;
    Complex<_Tp> *          x;
    Complex<_Tp> *          y;
    cudaStream_t            stream;
    cublasHandle_t          handle;

    ComplexAxpbyArgument(
      int                     single_vec_len,
      int                     stride    = 1,
      Complex<_Tp>            a         = 1,
      Complex<_Tp>            b         = 1,
      Complex<_Tp> *          x         = nullptr,
      Complex<_Tp> *          y         = nullptr,
      cudaStream_t            stream    = nullptr,
      cublasHandle_t          handle    = nullptr
    ) : single_vec_len(single_vec_len),
        stride(stride),
        a(a),
        b(b),
        x(x),
        y(y),
        stream(stream),
        handle(handle) {}
  };

  // methods
  void operator () (ComplexAxpbyArgument);
};



}
