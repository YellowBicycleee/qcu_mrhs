#pragma once
#include "complex/qcu_complex.cuh"
#include <cublas_v2.h>
#include <type_traits>

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

template <typename _Tp, typename = void> // only support for _Tp = Complex<float, double, half>
struct Complex_axpby;                    // result = aX + bY, (X, Y are complex vectors)

template <typename _Float>
struct Complex_axpby<_Float, std::enable_if_t <std::is_same_v<_Float, float>  ||
                                               std::is_same_v<_Float, double> ||
                                               std::is_same_v<_Float, half> > > 
{    
  // Argument type
  struct Complex_axpbyArgument {
    // start_idx 不在外部赋予，而是在内部赋予
    int               single_vec_len;
    int               inc_idx;
    Complex<_Float>*  res;
    Complex<_Float>*  x;
    Complex<_Float>*  y;
    Complex<_Float>*  a;
    Complex<_Float>*  b;
    cudaStream_t      stream;

    Complex_axpbyArgument(
      Complex<_Float>* res,
      Complex<_Float>* a, 
      Complex<_Float>* x,
      Complex<_Float>* b,
      Complex<_Float>* y,
      int              single_vec_len,
      int              inc_idx,
      cudaStream_t     stream
    ) : res(res),
        a(a),
        x(x),
        b(b),
        y(y),
        single_vec_len(single_vec_len),
        inc_idx(inc_idx),
        stream(stream) {}
  };

  // methods
  void operator () (Complex_axpbyArgument);
};



}
