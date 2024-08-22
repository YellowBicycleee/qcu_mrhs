#pragma once

#include "desc/qcu_desc.h"
#include "qcd/qcu_dslash.h"
#include "qcu_enum.h"
#include "qcu_float_float2_wrapper.h"
#include "qcu_blas/qcu_blas.h"
#include "qcd/qcu_dslash.h"
#include <cublas_v2.h>
namespace qcu::solver {

// OutputPrecision     既表示输入又表示输出精度，
// IteratePrecision    表示迭代精度
template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION InputPrecision,
          QCU_PRECISION IteratePrecision>
class BiCGStab {
public:
  BiCGStab () = delete;
  BiCGStab (DslashParam* param) : exterior_dslashParam_(param) {}
  ~BiCGStab() noexcept = default;
  bool solve();  // return true if converged

private:
  using OutputFloat  = typename qcu::Float2WrapperFromPrecision<OutputPrecision>::Float;
  using OutputFloat2 = typename qcu::Float2_t<OutputFloat>;
  using InputFloat   = typename qcu::Float2WrapperFromPrecision<InputPrecision>::Float;
  using InputFloat2  = typename qcu::Float2_t<InputFloat>;
  using IterateFloat = typename qcu::Float2WrapperFromPrecision<IteratePrecision>::Float;
  using IterateFloat2= typename qcu::Float2_t<IterateFloat>;
  
  struct InteriorOperator {
    // operator
    // 复数内积运算符
    using IterDotc           = typename qcu::qcu_blas::ComplexDotc<OutputFloat, IterateFloat>;
    using OutputDotc         = typename qcu::qcu_blas::ComplexDotc<OutputFloat, OutputFloat>;
    using IterDotcArgument   = typename IterDotc::DotcArgument;
    using OutputDotcArgument = typename OutputDotc::DotcArgument;
    IterDotc    iter_dotc;              // 迭代 内积运算符
    OutputDotc  output_dotc;    // 双精度内积运算符
  
    // norm2 运算符
    using IterNorm           = typename qcu::qcu_blas::ComplexNorm<OutputFloat, IterateFloat>;
    using OutputNorm         = typename qcu::qcu_blas::ComplexNorm<OutputFloat, OutputFloat>;
    using IterNormArgument   = typename IterNorm::ComplexNormArgument;
    using OutputNormArgument = typename OutputNorm::ComplexNormArgument;
    IterNorm    iter_norm;              // 迭代  norm2 运算符
    OutputNorm  output_norm;    // 高精度norm2 运算符
  
    // xpay 运算符
    using Iter_xpay           = typename qcu::qcu_blas::Complex_xpay<IterateFloat>::Complex_xpayArgument;
    using Output_xpay         = typename qcu::qcu_blas::Complex_xpay<OutputFloat>::Complex_xpayArgument;
    using Iter_xpayArgument   = typename Iter_xpay::Complex_xpayArgument;
    using Output_xpayArgument = typename Output_xpay::Complex_xpayArgument;
    Iter_xpay   iter_xpay;              // 迭代 xpay 运算符
    Output_xpay output_xpay;    // 高精度 xpay 运算符
  
    // xsay 运算符
    template <typename _Float> using Complex_xsay = typename qcu::qcu_blas::Complex_xsay<_Float>;
    using IterXsayArgument   = typename Complex_xsay<InputFloat>::Complex_xsayArgument;
    using OutputXsayArgument = typename Complex_xsay<OutputFloat>::Complex_xsayArgument;
    Complex_xsay<InputFloat>   iter_xsay;              // 迭代 xsay 运算符
    Complex_xsay<OutputFloat>  output_xsay;    // 高精度 xsay 运算符
  
    // axpby运算符
    template <typename _Float> using Complex_axpby = typename qcu::qcu_blas::Complex_axpby<_Float>;
    using IterAxpbyArgument   = typename Complex_axpby<InputFloat>::Complex_axpbyArgument;
    using OutputAxpbyArgument = typename Complex_axpby<OutputFloat>::Complex_axpbyArgument;
    Complex_axpby<InputFloat>   iter_axpby;             // 迭代 axpby 运算符
    Complex_axpby<OutputFloat>  output_axpby;   // 高精度 axpby 运算符

    // axpbypcz运算符
    template <typename _Float>
    using Complex_axpbypcz       = typename qcu::qcu_blas::Complex_axpbypcz<_Float>;
    using IterAxpbypczArgument   = typename Complex_axpbypcz<IterateFloat>::Complex_axpbypczArgument;
    using OutputAxpbypczArgument = typename Complex_axpbypcz<OutputFloat>::Complex_axpbypczArgument;
    Complex_axpbypcz<IterateFloat> iter_axpbypcz;   // 迭代 axpbypcz 运算符
    Complex_axpbypcz<OutputFloat>  output_axpbypcz; // 高精度 axpbypcz 运算符

    // elementwise_div 运算符
    template <typename _Tp>
    using ElementwiseDiv                   = typename qcu::qcu_blas::ElementwiseDiv<_Tp>;
    using IterElementwiseDivArgument       = typename ElementwiseDiv<Complex<IterateFloat>>::ElementwiseDivArgument;
    using OutputElementWiseDivArgument     = typename ElementwiseDiv<Complex<OutputFloat>>::ElementwiseDivArgument;
    ElementwiseDiv<Complex<IterateFloat>>  iter_elementwise_div;   // 迭代 elementwise_div 运算符
    ElementwiseDiv<Complex<OutputFloat>>   output_elementwise_div;   // 迭代 elementwise_div 运算符

    // elementwise_init 运算符
    template <typename _Tp> using ElementwiseInit = typename qcu::qcu_blas::ElementwiseInit<_Tp>;
    using IterElementwiseInitArgument    = typename ElementwiseInit<Complex<IterateFloat>>::ElementwiseInitArgument;
    using OutputElementwiseInitArgument  = typename ElementwiseInit<Complex<OutputFloat>>::ElementwiseInitArgument;
    ElementwiseInit<Complex<IterateFloat>>   iter_elementwise_init;           // 迭代 elementwise_init 运算符
    ElementwiseInit<Complex<OutputFloat>> output_elementwise_init; // 高精度 elementwise_init 运算符
  };



  // private functions:
  bool  tempBufferAllocate();
  void  tempBufferFree();
  void* reCalculate_b_even ();

  // member variables
  static constexpr int MaxTmpFermion_ = 6;              // 临时buffer的个数
  static constexpr int MaxOutputPrecisionFermion_ = 2;  // 输出精度的fermion个数
  int          maxIteration_     = 1000; // 最大迭代次数
  int          currentIteration_ = 0; // 当前迭代次数
  OutputFloat  maxPrec_          = 1e-6;

  DslashParam* exterior_dslashParam_ = nullptr;  // 从外部传入的参数，除了EVEN_ODD都能用
  Dslash*      dslash_operator_      = nullptr;

  // 计算中间需要的临时buffer
  bool  bufferAllocated_    = false;   // 内存是否已经分配
  void* tmpReduceMem_       = nullptr;
  void* new_b_iter_prec_    = nullptr; // 计算得到新的b，用于BICGSTAB的新b
  void* new_b_output_prec_  = nullptr; // 计算得到新的b，用于BICGSTAB的新b
  void* outputBuffer_[MaxOutputPrecisionFermion_]; // 输出精度buffer
  void* tmpFermionMrhs_[MaxTmpFermion_];  // 迭代精度buffer

  void* iter_scala_array_[3];
  void* output_scala_array_[5]; // 0存放kappa，在预处理的时候放入, 
                                // 1存放Complex(1, 0)

  void* alpha_array;
  void* beta_array;
  void* omega_array;

  cublasHandle_t cublasHandle_;
  // operator
  InteriorOperator interior_operator_;
};

}  // namespace qcu::solver