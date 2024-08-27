#include <cuda_fp16.h>
#include <thrust/system/cuda/detail/par.h>

#include "data_format/qcu_data_format_shift.cuh"
#include "solver/bicgstab.cuh"
namespace qcu::solver {

// 申请临时空间
template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION IteratePrecision>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::tempBufferAllocate () {
  if (bufferAllocated_) {
    return true;
  }

  const int Lx  = param_.lattDesc->X();
  const int Ly  = param_.lattDesc->Y();
  const int Lz  = param_.lattDesc->Z();
  const int Lt  = param_.lattDesc->T();
  const int vol = Lx * Ly * Lz * Lt;
  const int complex_vec_len = param_.mInput * param_.nColor * Ns; // on single point

  int iterate_float_size;
  int output_float_size;
  if      constexpr (IteratePrecision == QCU_HALF_PRECISION)   { iterate_float_size = sizeof (half);   }
  else if constexpr (IteratePrecision == QCU_SINGLE_PRECISION) { iterate_float_size = sizeof (float);  }
  else if constexpr (IteratePrecision == QCU_DOUBLE_PRECISION) { iterate_float_size = sizeof (double); }
  else                                                         { return false; }

  if      constexpr (OutputPrecision == QCU_HALF_PRECISION)    { output_float_size = sizeof (half);   }
  else if constexpr (OutputPrecision == QCU_SINGLE_PRECISION)  { output_float_size = sizeof (float);  }
  else if constexpr (OutputPrecision == QCU_DOUBLE_PRECISION)  { output_float_size = sizeof (double); }
  else                                                         { return false; }

  CHECK_CUDA(cudaMalloc(&tmpReduceMem_,      vol * complex_vec_len * output_float_size  * 2));
  CHECK_CUDA(cudaMalloc(&new_b_iter_prec_,   vol * complex_vec_len * iterate_float_size * 2));
  CHECK_CUDA(cudaMalloc(&new_b_output_prec_, vol * complex_vec_len * output_float_size  * 2));
  CHECK_CUDA(cudaMalloc(&result_x_output_prec_, vol * complex_vec_len * output_float_size * 2)); // full-length
  for (auto& buffer : outputBuffer_) {
    CHECK_CUDA(cudaMalloc(&buffer, vol / 2 * complex_vec_len * output_float_size  * 2)); // half vol
  }

  for (auto& buffer : tmpFermionMrhs_) {
    CHECK_CUDA(cudaMalloc(&buffer, vol / 2 * complex_vec_len * iterate_float_size * 2)); // half-vol
  }

  for (auto& buffer : iter_scala_array_) {
    CHECK_CUDA(cudaMalloc(&buffer, param_.mInput * iterate_float_size * 2));  // length: m-rhs * complex(2)
  }

  for (auto& buffer : output_scala_array_) {
    CHECK_CUDA(cudaMalloc(&buffer, param_.mInput * output_float_size * 2)); // length: m-rhs * complex(2)
  }

  CHECK_CUDA(cudaMalloc(&alpha_array, param_.mInput * output_float_size * 2));
  CHECK_CUDA(cudaMalloc(&beta_array,  param_.mInput * output_float_size * 2));
  CHECK_CUDA(cudaMalloc(&omega_array, param_.mInput * output_float_size * 2));

  // dslash运算符构造
  dslash_operator_ = new WilsonDslash(nullptr);
  // cublasHandler申请
  if (const cublasStatus_t stat = cublasCreate(&cublasHandle_); stat != CUBLAS_STATUS_SUCCESS) {
    printf("IN file %s, line %d, error happened\n", __FILE__, __LINE__);
    abort();
  }

  // 初始化kappa序列和1序列
  // init output_scala_array_[0] with Complex(kappa, 0) and init  output_scala_array_[1] with Complex(1, 0)
  // both with output precision
  void* output_prec_kappa = output_scala_array_[0];
  void* output_prec_ones  = output_scala_array_[1];
  void* output_prec_kappa_square = output_scala_array_[5];

  // init multiple-Mrhs kappa s
  using InitArgument = typename InteriorOperator::template ElementwiseInit<Complex<OutputFloat>>::ElementwiseInitArgument;
  InitArgument 
    output_elementwise_init_arg (
          static_cast<Complex<OutputFloat>*>(output_prec_kappa),
          Complex<OutputFloat>{static_cast<OutputFloat>(param_.kappa), 0},
          param_.mInput, param_.stream1
    );
  interior_operator_.output_elementwise_init(output_elementwise_init_arg);


  // init mrhs 1 s
  output_elementwise_init_arg.res = static_cast<Complex<OutputFloat>*>(output_prec_ones);
  output_elementwise_init_arg.val = Complex<OutputFloat>{1, 0};
  interior_operator_.output_elementwise_init(output_elementwise_init_arg);
  // sync
  // init mrhs kappa * kappa s
  output_elementwise_init_arg.res = static_cast<Complex<OutputFloat>*>(output_prec_kappa_square);
  output_elementwise_init_arg.val = Complex<OutputFloat>{static_cast<OutputFloat>(param_.kappa)
                                                        * static_cast<OutputFloat>(param_.kappa), 0};
  interior_operator_.output_elementwise_init(output_elementwise_init_arg);

  CHECK_CUDA(cudaStreamSynchronize(param_.stream1));
  bufferAllocated_ = true;

  return true;
}

// 释放临时空间
// 申请临时空间
template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION IteratePrecision>
void BiCGStabImpl<OutputPrecision,IteratePrecision>::tempBufferFree() {
  CHECK_CUDA(cudaFree(tmpReduceMem_));
  CHECK_CUDA(cudaFree(new_b_iter_prec_));
  CHECK_CUDA(cudaFree(new_b_output_prec_));
  CHECK_CUDA(cudaFree(result_x_output_prec_));
  for (auto& buffer : outputBuffer_) {
    CHECK_CUDA(cudaFree(buffer));
    buffer = nullptr;
  }
  for (auto& buffer : tmpFermionMrhs_) {
    CHECK_CUDA(cudaFree(buffer));
    buffer = nullptr;
  }
  for (auto& buffer : iter_scala_array_) {
    CHECK_CUDA(cudaFree(buffer));
    buffer = nullptr;
  }
  for (auto& buffer : output_scala_array_) {
    CHECK_CUDA(cudaFree(buffer));
    buffer = nullptr;
  }
  CHECK_CUDA(cudaFree(alpha_array));
  CHECK_CUDA(cudaFree(beta_array));
  CHECK_CUDA(cudaFree(omega_array));
  if (const cublasStatus_t stat = cublasDestroy(cublasHandle_); stat != CUBLAS_STATUS_SUCCESS) {
    printf("IN file %s, line %d, error happened\n", __FILE__, __LINE__);
    abort();
  }

  if (dslash_operator_) {
    delete dslash_operator_;
  }
  bufferAllocated_ = false;
}


template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION IteratePrecision>
void* BiCGStabImpl<OutputPrecision, IteratePrecision>::reCalculate_b_even () {
  const int Lx  = param_.lattDesc->X();
  const int Ly  = param_.lattDesc->Y();
  const int Lz  = param_.lattDesc->Z();
  const int Lt  = param_.lattDesc->T();
  const int vol = Lx * Ly * Lz * Lt;
  const int single_vec_len = param_.nColor * Ns;
  const int complex_vec_len = param_.mInput * single_vec_len;

  // copy origin_even_b to new_b_ first, with correct precision
  void* origin_even_b = param_.input_b_mrhs;
  void* origin_odd_b  = static_cast<Complex<OutputFloat>*>(origin_even_b) + vol / 2 * complex_vec_len;
  // then, regenerate new_b on right hand side of equation. by new_even_b = \kappa D_{oe} b_{e} + b_{o}
  void* new_even_b = new_b_output_prec_;

  DslashParam param {
    /*param.daggerFlag      */   false,
    /*param.precision       */   OutputPrecision,
    /*param.nColor          */   param_.nColor,
    /*param.mInput          */   param_.mInput,
    /*param.parity          */   ODD_PARITY,      // D_{oe} b_{e}
    /*param.kappa           */   param_.kappa,
    /*param.fermionIn_MRHS  */   origin_even_b,   // b_{e}
    /*param.fermionOut_MRHS */   new_even_b,      // new_even_b
    /*param.gauge           */   param_.gauge,
    /*param.lattDesc        */   param_.lattDesc,
    /*param.procDesc        */   param_.procDesc,
    /*param.stream1         */   param_.stream1,
    /*param.stream2         */   param_.stream2,
  };

  dslash_operator_->setParam(&param);
  dslash_operator_->apply();  // new_even_b = D_{oe} b_{e}

  void* output_prec_kappa = output_scala_array_[0]; // this array stores kappa

  // batch new_b{e} = b_{o} + kappa D_{oe} b_{e}
  using XpayArgument = typename InteriorOperator::template Complex_xpay<OutputFloat>::Complex_xpayArgument;
  XpayArgument output_xpay_arg (
    static_cast<Complex<OutputFloat>*>(new_even_b),        // res = new_even_b = x + ay = x + kappa new_even_b
    static_cast<Complex<OutputFloat>*>(origin_odd_b),      // x = origin_odd_b
    static_cast<Complex<OutputFloat>*>(output_prec_kappa), // a = output_prec_kappa
    static_cast<Complex<OutputFloat>*>(new_even_b),        // y = new_even_b = D_{oe} b_{e}
    single_vec_len * vol / 2,
    param.mInput,
    param.stream1
  );
  interior_operator_.output_xpay(output_xpay_arg);
  return new_even_b;
}
// donnot use HALF to be the output precision
template class BiCGStabImpl<QCU_DOUBLE_PRECISION, QCU_DOUBLE_PRECISION>;
template class BiCGStabImpl<QCU_DOUBLE_PRECISION, QCU_SINGLE_PRECISION>;
template class BiCGStabImpl<QCU_DOUBLE_PRECISION, QCU_HALF_PRECISION>;
template class BiCGStabImpl<QCU_SINGLE_PRECISION, QCU_DOUBLE_PRECISION>;
template class BiCGStabImpl<QCU_SINGLE_PRECISION, QCU_SINGLE_PRECISION>;
template class BiCGStabImpl<QCU_SINGLE_PRECISION, QCU_HALF_PRECISION>;
}