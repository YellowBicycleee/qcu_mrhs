#pragma once
#include "complex/qcu_complex.cuh"
#include <type_traits>
#include <cublas_v2.h>

namespace qcu::qcu_blas {

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

}
