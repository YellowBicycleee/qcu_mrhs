#pragma once
#include "complex/qcu_complex.cuh"
namespace qcu::qcu_blas {

struct ReductionParam {
  int single_vector_length;
  int m_rhs;
  int stride;
  void* tmpBuffer;  // reduce结果一般都很短，tmpBuffer作为第一次的输出，第二次的输入+输出
  void* input1;
  void* input2;
  void** lastRes;  // device array
  cudaStream_t stream;

  ReductionParam(
      int p_single_vector_length,
      int p_m_rhs,
      int p_stride,
      void* p_tmpBuffer = nullptr,
      void* p_input1 = nullptr,
      void* p_input2 = nullptr,
      void** p_lastRes = nullptr,
      cudaStream_t p_stream = NULL
    ) noexcept 
      : single_vector_length (p_single_vector_length)
      , m_rhs (p_m_rhs)
      , stride (p_stride)
      , tmpBuffer (p_tmpBuffer)
      , input1 (p_input1)
      , input2 (p_input2)
      , lastRes (p_lastRes)
      , stream (p_stream) 
  {}
};


template <typename OutputFloat, typename InputFloat>
struct ReductionNorm {
  void operator () (ReductionParam);
};

template <typename OutputFloat, typename InputFloat>
struct ReductionInnerprod {
  void operator () (ReductionParam);
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