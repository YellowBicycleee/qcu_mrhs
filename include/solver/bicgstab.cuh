#pragma once

#include <cublas_v2.h>
#include "base/datatype/qcu_float2.cuh"
#include "desc/qcu_desc.h"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_blas/qcu_blas.h"
#include "qcu_public.h"

namespace qcu::solver {

struct BiCGStabParam {
  int nColor;
  int mInput;
  double kappa;
  void* output_x_mrhs;
  void* input_b_mrhs;
  void* gauge;
  const QcuLattDesc* lattDesc;
  const QcuProcDesc* procDesc;
  cudaStream_t stream1;
  cudaStream_t stream2;
};

// OutputPrecision     既表示输入又表示输出精度，
// IteratePrecision    表示迭代精度
template <QcuPrecision OutputPrecision,
          QcuPrecision IteratePrecision>
class BiCGStabImpl {
public:
  BiCGStabImpl () = delete;
  BiCGStabImpl (BiCGStabParam& param, int max_iteration = 1000,
                double max_precision = 1e-6)
          :
            param_(param),
            maxIteration_(max_iteration),
            maxPrec_(OutputFloat(double(max_precision)))
  {
    tempBufferAllocate();
  }
  ~BiCGStabImpl() noexcept {
    tempBufferFree();
  }
  bool solve();  // return true if converged
private:
  bool solve_odd();
  bool solve_odd_policy1(); // 单独计算norm和内积
  bool solve_odd_policy2(); // 所有残差按一个计算
  bool solve_even();
  using OutputFloat  = typename qcu::Float2WrapperFromPrecision<OutputPrecision>::Float;
  using OutputFloat2 = typename qcu::Float2_t<OutputFloat>;
  using IterateFloat = typename qcu::Float2WrapperFromPrecision<IteratePrecision>::Float;
  using IterateFloat2= typename qcu::Float2_t<IterateFloat>;
  
  struct InteriorOperator {
    // operator
    // 复数内积运算符
    template <typename _OutputFloat, typename _InputFloat>
    using ComplexDotc         = typename qcu::qcu_blas::ComplexDotc<_OutputFloat, _InputFloat>;
    ComplexDotc<OutputFloat, IterateFloat> iter_dotc;    // 迭代 内积运算符
    ComplexDotc<OutputFloat, OutputFloat> output_dotc;  // 双精度内积运算符

    // norm2 运算符
    template <typename _OutputFloat, typename _InputFloat>
    using ComplexNorm        = typename qcu::qcu_blas::ComplexNorm<_OutputFloat, _InputFloat>;
    ComplexNorm<OutputFloat, IterateFloat>  iter_norm;      // 迭代  norm2 运算符
    ComplexNorm<OutputFloat, OutputFloat>  output_norm;    // 高精度norm2 运算符
  
    // xpay 运算符
    template<typename _Float>
    using Complex_xpay        = typename qcu::qcu_blas::Complex_xpay<_Float>;
    Complex_xpay<IterateFloat> iter_xpay;              // 迭代 xpay 运算符
    Complex_xpay<OutputFloat>  output_xpay;    // 高精度 xpay 运算符
  
    // xsay 运算符
    template <typename _Float> 
    using Complex_xsay = typename qcu::qcu_blas::Complex_xsay<_Float>;
    Complex_xsay<IterateFloat>   iter_xsay;              // 迭代 xsay 运算符
    Complex_xsay<OutputFloat>  output_xsay;    // 高精度 xsay 运算符
  
    // axpby运算符
    template <typename _Float> 
    using Complex_axpby = typename qcu::qcu_blas::Complex_axpby<_Float>;
    Complex_axpby<IterateFloat>   iter_axpby;             // 迭代 axpby 运算符
    Complex_axpby<OutputFloat>  output_axpby;   // 高精度 axpby 运算符

    // axpbypcz运算符
    template <typename _Float>
    using Complex_axpbypcz       = typename qcu::qcu_blas::Complex_axpbypcz<_Float>;
    Complex_axpbypcz<IterateFloat> iter_axpbypcz;   // 迭代 axpbypcz 运算符
    Complex_axpbypcz<OutputFloat>  output_axpbypcz; // 高精度 axpbypcz 运算符

    // elementwise_div 运算符
    template <typename _Tp>
    using ElementwiseDiv                   = typename qcu::qcu_blas::ElementwiseDiv<_Tp>;
    ElementwiseDiv<Complex<IterateFloat>>  iter_elementwise_div;   // 迭代 elementwise_div 运算符
    ElementwiseDiv<Complex<OutputFloat>>   output_elementwise_div;   // 迭代 elementwise_div 运算符

    // elementwise_mul 运算符
    template <typename _Tp>
    using ElementwiseMul                   = typename qcu::qcu_blas::ElementwiseMul<_Tp>;
    ElementwiseMul<Complex<IterateFloat>>  iter_elementwise_mul;   // 迭代 elementwise_div 运算符
    ElementwiseMul<Complex<OutputFloat>>   output_elementwise_mul;   // 迭代 elementwise_div 运算符

    // elementwise_init 运算符
    template <typename _Tp> using ElementwiseInit = typename qcu::qcu_blas::ElementwiseInit<_Tp>;
    ElementwiseInit<Complex<IterateFloat>> iter_elementwise_init;           // 迭代 elementwise_init 运算符
    ElementwiseInit<Complex<OutputFloat>>  output_elementwise_init; // 高精度 elementwise_init 运算符

    InteriorOperator() = default;
  };

  // private functions:
  bool  tempBufferAllocate();
  void  tempBufferFree();
  void* reCalculate_b_even ();

  // member variables
  static constexpr int MaxTmpFermion_             = 6;              // 临时buffer的个数
  static constexpr int MaxOutputPrecisionFermion_ = 12;  // 输出精度的fermion个数
  int          maxIteration_     = 1000; // 最大迭代次数
  int          currentIteration_ = 0; // 当前迭代次数
  OutputFloat  maxPrec_          = 1e-6;

  // Dslash*      dslash_operator_oe_      = nullptr;
  Dslash*      dslash_operator_      = nullptr;
  // Dslash*      dslash_operator_eo_   = nullptr;

  // 计算中间需要的临时buffer
  bool  bufferAllocated_      = false;   // 内存是否已经分配
  void* tmpReduceMem_         = nullptr;
  void* new_b_iter_prec_      = nullptr;           // 计算得到新的b，用于BICGSTAB的新b
  void* new_b_output_prec_    = nullptr;           // 计算得到新的b，用于BICGSTAB的新b
  void* result_x_output_prec_ = nullptr;           // 用于存储迭代得到的x
  void* outputBuffer_[MaxOutputPrecisionFermion_]; // 输出精度buffer
  void* tmpFermionMrhs_[MaxTmpFermion_];           // 迭代精度buffer

  void* iter_scala_array_[3];
  void* output_scala_array_[9]; // [0]存放Complex(kappa, 0)，[1]存放Complex(1, 0) [5]存放Complex(kappa * kappa, 0)


  void* alpha_array;
  void* beta_array;
  void* omega_array;

  cublasHandle_t cublasHandle_;
  // operator
  InteriorOperator interior_operator_;
  BiCGStabParam& param_;
};

void ApplyBicgStab (BiCGStabParam& param,  QcuPrecision outputPrecision, QcuPrecision iteratePrecision,
                    int max_iteration = 1000, double max_precision = 1e-6);
}  // namespace qcu::solver