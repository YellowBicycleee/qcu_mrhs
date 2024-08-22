#include "solver/bicgstab.cuh"
#include <cuda_fp16.h>
#include "data_format/qcu_data_format_shift.cuh"
namespace qcu::solver {

// 申请临时空间
template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION InputPrecision,
          QCU_PRECISION IteratePrecision>
bool BiCGStab<OutputPrecision, InputPrecision, IteratePrecision>::tempBufferAllocate () {
  if (bufferAllocated_) {
    return true;
  }

  const int Lx = exterior_dslashParam_->lattDesc->X();
  const int Ly = exterior_dslashParam_->lattDesc->Y();
  const int Lz = exterior_dslashParam_->lattDesc->Z();
  const int Lt = exterior_dslashParam_->lattDesc->T();
  const int vol = Lx * Ly * Lz * Lt;
  const int complex_vec_len =   exterior_dslashParam_->mInput 
                              * exterior_dslashParam_->nColor 
                              * Ns; // on single point

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
  for (auto& buffer : outputBuffer_) {
    CHECK_CUDA(cudaMalloc(&buffer, vol * complex_vec_len * output_float_size  * 2));
  }

  for (auto& buffer : tmpFermionMrhs_) {
    CHECK_CUDA(cudaMalloc(&buffer, vol * complex_vec_len * iterate_float_size * 2));
  }  
  // void* iter_scala_array_[3];
  // void* output_scala_array_[3];

  for (auto& buffer : iter_scala_array_) {
    CHECK_CUDA(cudaMalloc(&buffer, exterior_dslashParam_->mInput * iterate_float_size * 2));
  }

  for (auto& buffer : output_scala_array_) {
    CHECK_CUDA(cudaMalloc(&buffer, exterior_dslashParam_->mInput * output_float_size * 2));
  }

  CHECK_CUDA(cudaMalloc(&alpha_array, exterior_dslashParam_->mInput * output_float_size * 2));
  CHECK_CUDA(cudaMalloc(&beta_array,  exterior_dslashParam_->mInput * output_float_size * 2));
  CHECK_CUDA(cudaMalloc(&omega_array, exterior_dslashParam_->mInput * output_float_size * 2));

  // dslash运算符构造
  dslash_operator_ = new WilsonDslash(exterior_dslashParam_);

  // cublasHandler申请
  cublasStatus_t stat = cublasCreate(&cublasHandle_);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("IN file %s, line %d, error happened\n", __FILE__, __LINE__);
    abort();
  }

  // 初始化kappa序列和1序列
  void* output_prec_kappa = output_scala_array_[0];

  // init mrhs kappa s
  using InitArgument = typename InteriorOperator::ElementwiseInit<Complex<OutputFloat>>::ElementwiseInitArgument;
  InitArgument 
    output_elementwise_init_arg (
          reinterpret_cast<Complex<OutputFloat>*>(output_prec_kappa),
          Complex<OutputFloat>{exterior_dslashParam_->kappa, 0},
          exterior_dslashParam_->mInput,
          exterior_dslashParam_->stream1
    );
  interior_operator_.output_elementwise_init(output_elementwise_init_arg);

  // init mrhs 1 s
  output_elementwise_init_arg.res = reinterpret_cast<Complex<OutputFloat>*>(output_scala_array_[1]);
  output_elementwise_init_arg.val = Complex<OutputFloat>{1, 0};
  interior_operator_.output_elementwise_init(output_elementwise_init_arg);
  // sync
  CHECK_CUDA(cudaStreamSynchronize(exterior_dslashParam_->stream1));
  bufferAllocated_ = true;

  return true;
}

// 释放临时空间
// 申请临时空间
template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION InputPrecision,
          QCU_PRECISION IteratePrecision>
void BiCGStab<OutputPrecision, InputPrecision,IteratePrecision>::tempBufferFree() {
  CHECK_CUDA(cudaFree(tmpReduceMem_));
  CHECK_CUDA(cudaFree(new_b_iter_prec_));
  CHECK_CUDA(cudaFree(new_b_output_prec_));
  for (auto& buffer : outputBuffer_) {
    CHECK_CUDA(cudaFree(buffer));
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
  cublasStatus_t stat = cublasDestroy(cublasHandle_);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("IN file %s, line %d, error happened\n", __FILE__, __LINE__);
    abort();
  }
  
  bufferAllocated_ = false;
}


template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION InputPrecision,
          QCU_PRECISION IteratePrecision>
void* BiCGStab<OutputPrecision, InputPrecision, IteratePrecision>::reCalculate_b_even () {
  const int Lx = exterior_dslashParam_->lattDesc->X();
  const int Ly = exterior_dslashParam_->lattDesc->Y();
  const int Lz = exterior_dslashParam_->lattDesc->Z();
  const int Lt = exterior_dslashParam_->lattDesc->T();
  const int vol = Lx * Ly * Lz * Lt;
  const int complex_vec_len =   exterior_dslashParam_->mInput 
                              * exterior_dslashParam_->nColor
                              * Ns;

  // copy origin_even_b to new_b_ first
  copyComplexVector_interface(outputBuffer_[0], OutputPrecision, 
                              exterior_dslashParam_->fermionIn_MRHS,  
                              InputPrecision, 
                              vol * complex_vec_len, 
                              exterior_dslashParam_->stream1);
  void* origin_even_b = outputBuffer_[0];
  void* origin_odd_b  = static_cast<OutputFloat2*>(origin_even_b) + vol / 2 * complex_vec_len;

  void* new_even_b = new_b_output_prec_;
  void* new_odd_b  = static_cast<OutputFloat2*>(new_b_output_prec_) + vol / 2 * complex_vec_len;

  DslashParam param {
    /*param.daggerFlag      = */   false,
    /*param.precision       = */   OutputPrecision,
    /*param.nColor          = */   exterior_dslashParam_->nColor, // param.nColor;
    /*param.mInput          = */   exterior_dslashParam_->mInput, // param.mInput;
    /*param.parity          = */   ODD_PARITY,      // D_{oe} b_{e}
    /*param.kappa           = */   exterior_dslashParam_->kappa, // param.kappa;
    /*param.fermionIn_MRHS  = */   origin_even_b,
    /*param.fermionOut_MRHS = */   new_odd_b,
    /*param.gauge           = */   exterior_dslashParam_->gauge,
    /*param.lattDesc        = */   exterior_dslashParam_->lattDesc,
    /*param.procDesc        = */   exterior_dslashParam_->procDesc,
    /*param.stream1         = */   exterior_dslashParam_->stream1,
    /*param.stream2         = */   exterior_dslashParam_->stream2,
  };  

  dslash_operator_->setParam(&param);
  dslash_operator_->apply();  // new_odd_b = D_{oe} b_{e}

  void* output_prec_kappa = output_scala_array_[0];

  // // init mrhs kappa s
  // using InitArgument = typename InteriorOperator::OutputElementwiseInitArgument;
  // InitArgument 
  //   output_elementwise_init_arg (
  //         reinterpret_cast<Complex<OutputFloat>*>(output_prec_kappa),
  //         Complex<OutputFloat>{param.kappa, 0},
  //         exterior_dslashParam_->mInput,
  //         param.stream1
  // );
  // interior_operator_.output_elementwise_init(output_elementwise_init_arg);

  // batch new_b{e} = b_{o} + kappa D_{oe} b_{e}
  using XpbyArgument = typename InteriorOperator::Complex_xpay<OutputFloat>::Complex_xpayArgument;
  XpbyArgument output_xpby_arg (
    reinterpret_cast<Complex<OutputFloat>*>(new_even_b),        // res = new_even_b = x + a y
    reinterpret_cast<Complex<OutputFloat>*>(origin_odd_b),      // x = origin_odd_b
    reinterpret_cast<Complex<OutputFloat>*>(output_prec_kappa), // a = output_prec_kappa
    reinterpret_cast<Complex<OutputFloat>*>(new_odd_b),         // y = new_odd_b = D_{oe} b_{e}
    complex_vec_len,
    param.mInput,
    param.stream1
  );
  return new_b_output_prec_;
}

template class BiCGStab<QCU_DOUBLE_PRECISION, QCU_DOUBLE_PRECISION, QCU_DOUBLE_PRECISION>; 
}