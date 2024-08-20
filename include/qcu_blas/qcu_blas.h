#pragma once
#include "complex/qcu_complex.cuh"
#include <cublas_v2.h>

namespace qcu::qcu_blas {

// struct ReductionParam {
//   int            single_vector_length;
//   int            stride    = 1;
//   void*          tmpBuffer;  // reduce结果一般都很短，tmpBuffer作为第一次的输出，第二次的输入+输出
//   void*          input1;
//   void*          input2;
//   void*          resArr;  // [device_res1, device_res2, ...]
//   cudaStream_t   stream;
//   cublasHandle_t handle;

//   ReductionParam(
//     int            single_vector_length,
//     int            stride = 1,
//     void*          tmpBuffer = nullptr,
//     void*          input1 = nullptr,
//     void*          input2 = nullptr,
//     void*          resArr = nullptr,
//     cudaStream_t   stream = nullptr,
//     cublasHandle_t handle = nullptr
//   ) : single_vector_length(single_vector_length),
//       stride(stride),
//       tmpBuffer(tmpBuffer),
//       input1(input1),
//       input2(input2),
//       resArr(resArr),
//       stream(stream),
//       handle(handle) {} 
// };


template <typename OutputFloat, typename InputFloat>
struct ComplexNorm {
  // Argument type
  struct ComplexNormArgument {
    int                     single_vector_length;
    int                     stride;
    OutputFloat         *   tmpBuffer;  // reduce结果一般都很短，tmpBuffer作为第一次的输出，第二次的输入+输出
                                        // when you use cublas, you can set it to nullptr
    Complex<InputFloat> *   input;
    OutputFloat         *   resArr;  // [device_res1, device_res2, ...]
    cudaStream_t            stream;
    cublasHandle_t          handle;

    ComplexNormArgument(
      int                     single_vector_length,
      int                     stride    = 1,
      OutputFloat         *   tmpBuffer = nullptr,
      Complex<InputFloat> *   input     = nullptr,
      OutputFloat         *   resArr    = nullptr,
      cudaStream_t            stream    = nullptr,
      cublasHandle_t          handle    = nullptr
    ) : single_vector_length(single_vector_length),
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
    int                     single_vector_length;
    int                     stride;
    Complex<OutputFloat>*   tmpBuffer;  // reduce结果一般都很短，tmpBuffer作为第一次的输出，第二次的输入+输出
                                        // when you use cublas, you can set it to nullptr
    Complex<InputFloat> *   input1;
    Complex<InputFloat> *   input2;
    Complex<OutputFloat>*   resArr;  // [device_res1, device_res2, ...]
    cudaStream_t            stream;
    cublasHandle_t          handle;

    DotcArgument(
      int                     single_vector_length,
      int                     stride    = 1,
      Complex<OutputFloat>*   tmpBuffer = nullptr,
      Complex<InputFloat> *   input1    = nullptr,
      Complex<InputFloat> *   input2    = nullptr,
      Complex<OutputFloat>*   resArr    = nullptr,
      cudaStream_t            stream    = nullptr,
      cublasHandle_t          handle    = nullptr
    ) : single_vector_length(single_vector_length),
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
struct ElementWiseParam {
  int m_rhs;
  int single_vector_length;
  void* res;
  Complex <_Tp> a;
  void* x;
  Complex <_Tp> b;
  void* y;
  Complex <_Tp> c;
  void* z;
};

template <typename _Tp>
struct QcuCax {
  void operator () (ElementWiseParam<_Tp>); 
};

template <typename _Tp>
struct QcuCaxpCby {   // Res = ax + by
  void operator () (ElementWiseParam<_Tp>);
};
template <typename _Tp>
// fused operator
struct QcuCaxpCbyCcz {  // Res = ax + by + cz
  void operator () (ElementWiseParam<_Tp>);
};


}
