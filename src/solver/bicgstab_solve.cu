#include "solver/bicgstab.cuh"
#include <vector>
#include <type_traits>
#include "data_format/qcu_data_format_shift.cuh"

namespace qcu::solver {
template <typename _Float, std::enable_if_t<std::is_same_v<_Float, float> ||
                                            std::is_same_v<_Float, double>>* = nullptr>
inline bool isConverged ( const std::vector<_Float>& norm_r_array, 
                          const std::vector<_Float>& norm_b_array,
                          _Float target_diff
) 
{
  // calculate the relative error
  _Float total_error = 0.0;

  int size = norm_r_array.size();
  for (int i = 0; i < size; i++) {
    total_error += norm_r_array[i];
  }
  return total_error / size < target_diff;
}

template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION InputPrecision,
          QCU_PRECISION IteratePrecision>
bool BiCGStab<OutputPrecision, InputPrecision, IteratePrecision>::solve_odd() {
  std::vector<OutputFloat> norm_r_array  (exterior_dslashParam_->mInput, 1.0);  // 使用accumulate计算误差
  std::vector<OutputFloat> norm_b_array(exterior_dslashParam_->mInput, 1.0);  // 计算b的模长
  // diff_array = [r1, r2, r3, ...] / [b1, b2, b3, ...]

  const int Lx     = exterior_dslashParam_->lattDesc->X();
  const int Ly     = exterior_dslashParam_->lattDesc->Y();
  const int Lz     = exterior_dslashParam_->lattDesc->Z();
  const int Lt     = exterior_dslashParam_->lattDesc->T();
  const int mInput = exterior_dslashParam_->mInput;
  const int vol = Lx * Ly * Lz * Lt;
  const int single_complex_vec_len = exterior_dslashParam_->nColor * Ns;
  const int complex_vec_len =   exterior_dslashParam_->mInput 
                              * exterior_dslashParam_->nColor 
                              * Ns; // on single point
  cudaStream_t stream1 = exterior_dslashParam_->stream1;
  cudaStream_t stream2 = exterior_dslashParam_->stream2;
  // prelogue
  // solve x_o
  // get new B
  reCalculate_b_even ();

  // calculate norm of b and store them
  using OutputNormArgument = typename InteriorOperator::OutputNormArgument;
  using IterNormArgument   = typename InteriorOperator::IterNormArgument;

  void* new_b_even_output_prec_ = new_b_output_prec_;
  void* output_new_b_even_norm = output_scala_array_[2];

  void* x_e          = outputBuffer_[9];
  void* Ap           = outputBuffer_[1];  // 每轮要留住 outputBuffer_[1]作为 Ap
  void* As           = outputBuffer_[2];  // 每轮要留住 outputBuffer_[2]作为 As
  void* s            = outputBuffer_[3];
  void* r_new        = outputBuffer_[4];
  void* reduceBuffer = outputBuffer_[5];
  void* x_new        = outputBuffer_[6];
  void* p_new        = outputBuffer_[7];
  void* x            = outputBuffer_[8]; 
  void* one_array    = output_scala_array_[1];
  void* kappa_array  = output_scala_array_[0];
    
  void* r_r0_norm     = output_scala_array_[2]; // 要用到最后
  void* Ap_p_norm     = output_scala_array_[3];
  void* r_new_r0_norm = output_scala_array_[3];
  void* As_s_norm     = output_scala_array_[3];
  void* As_As_norm    = output_scala_array_[4];

  OutputNormArgument output_norm_arg {
    vol * single_complex_vec_len / 2,                           // int single_vec_len;
    mInput,                                                     // int stride;
    static_cast<OutputFloat*>(outputBuffer_[0]),                // OutputFloat*   tmpBuffer 
    static_cast<Complex<OutputFloat>*>(new_b_even_output_prec_),// Complex<InputFloat> *   input;
    static_cast<OutputFloat*>(output_new_b_even_norm),          // OutputFloat*  resArr;
    stream1,
    cublasHandle_
  };

  interior_operator_.output_norm(output_norm_arg); // 计算norm，一次性保存到host端

  CHECK_CUDA(cudaMemcpy(norm_b_array.data(), 
                        output_new_b_even_norm,
                        sizeof(OutputFloat) * mInput, 
                        cudaMemcpyDeviceToHost,
                        stream1));
  // store norm of b in norm_b_array (host)
  CHECK_CUDA(cudaStreamSynchronize(stream1));

  // R = b - A * x = b - Dslash * x, x在可以初始化为0
  // 存放 Ax 到 outputBuffer_[1]
  DslashParam dslashParam {
    false,                                      // DslashParam(bool p_daggerFlag,
    OutputPrecision,                            // QCU_PRECISION p_precision,
    exterior_dslashParam_->nColor,              // int p_nColor, 
    exterior_dslashParam_->mInput,              // int p_mInput, 
    EVEN_PARITY,                                // int p_parity,
    OutputFloat(exterior_dslashParam_->kappa),  // double p_kappa, 
    new_b_even_output_prec_,                    // void* p_fermionIn_MRHS
    outputBuffer_[1],                           // void* p_fermionOut_MRHS,
    exterior_dslashParam_->gauge,               // void* p_gauge, 
    exterior_dslashParam_->lattDesc,            // const QcuLattDesc* p_lattDesc,
    exterior_dslashParam_->procDesc,            // const QcuProcDesc* p_procDesc,
    stream1,                                    // cudaStream_t p_stream1 = NULL,
    stream2                                     // cudaStream_t p_stream2 = NULL)
  };

  // dslash_operator_->setParam(&dslashParam);
  // dslash_operator_->apply();  // outputBuffer_[1] = Dslash * new_b_even_output_prec_
  // Ax存入outputBuffer_[1]，接下来计算r = b - Ax放入outputBuffer_[0]
  // r0 = b - Ax = b
  using Output_xsayArgument = typename InteriorOperator::Output_xsayAruArgument;
  Output_xsayArgument output_xsay_arg {
      // Complex_xsayArgument
      reinterpret_cast<Complex<OutputFloat>*>(outputBuffer_[0]),        // Complex<_Float>* res,
      reinterpret_cast<Complex<OutputFloat>*>(new_b_even_output_prec_), // Complex<_Float>* x,
      reinterpret_cast<Complex<OutputFloat>*>(output_scala_array_[1]),  // Complex<_Float>* a,
      reinterpret_cast<Complex<OutputFloat>*>(outputBuffer_[1]),        // Complex<_Float>* y,
      vol * single_complex_vec_len / 2,                                 // int single_vec_len,
      mInput,                                                           // int inc_idx,
      stream1
  };
  // interior_operator_.output_xsay(output_xsay_arg); // r0 = b - Ax, 存入outputBuffer_[0], outputBuffer_[1]空闲

  // begin iteration
  // InnerProduct Param
  using OutputDotcArgument = typename InteriorOperator::IterDotcArgument;
  OutputDotcArgument outputDotArg {
    vol * single_complex_vec_len / 2,                              // int single_vec_len;
    mInput,                                                     // int stride;
    nullptr,//reduceBuffer),      // Complex<OutputFloat>* tmpBuffer;  
    nullptr,//r),                 // Complex<InputFloat> *   input1;
    nullptr,//r0),                // Complex<InputFloat> *   input2;
    nullptr,//r_r0_norm),         // Complex<OutputFloat>*   resArr;
    stream1,                                                    // cudaStream_t stream;
    cublasHandle_,                                              // cublasHandle_t handle;
  };
  // ElementwiseDiv Param
  using OutputElementwiseDivArgument = typename InteriorOperator::OutputElementwiseDivArgument;
  OutputElementwiseDivArgument output_elementwise_div_arg {
    nullptr, // _Tp* res,
    nullptr,   // _Tp* x,
    nullptr,   // _Tp* y,
    mInput                                                // int  vec_len,
  };
  // ElementwiseMul Param
  using OutputElementwiseMulArgument = typename InteriorOperator::OutputElementwiseMulArgument;
  OutputElementwiseMulArgument output_elementwise_mul_arg {
    nullptr, // _Tp* res,
    nullptr,   // _Tp* x,
    nullptr,   // _Tp* y,
    mInput                                                // int  vec_len,
  };
  // 
  using OutputAxpbypczArgument = typename InteriorOperator::OutputAxpbypczArgument;
  OutputAxpbypczArgument output_axpbypcz_arg {
    nullptr,  //   Complex<_Float>* res,
    nullptr,  //   Complex<_Float>* a, 
    nullptr,  //   Complex<_Float>* x,
    nullptr,  //   Complex<_Float>* b,
    nullptr,  //   Complex<_Float>* y,
    nullptr,  //   Complex<_Float>* c,
    nullptr,  //   Complex<_Float>* z,
    vol * single_complex_vec_len / 2,   //   int              single_vec_len,
    mInput,  //   int              inc_idx,
    stream1  //   cudaStream_t     stream = nullptr
  };

  using Output_xpayArgument = typename InteriorOperator::Output_xpayAruArgument;
  Output_xpayArgument output_xpay_arg {
      // Complex_xsayArgument
      nullptr,        // Complex<_Float>* res,
      nullptr, // Complex<_Float>* x,
      nullptr,  // Complex<_Float>* a,
      nullptr,        // Complex<_Float>* y,
      vol * single_complex_vec_len / 2,                                 // int single_vec_len,
      mInput,                                                           // int inc_idx,
      stream1
  };

  void* r0 = outputBuffer_[0];
  void* r  = outputBuffer_[0]; // 初始时 r0 == r1  
  void* p  = outputBuffer_[0]; // p = r0，从此outputBuffer[1]不变，不能再使用
  // 开始迭代，达到最大迭代次数不收敛则返回false
  for (currentIteration_ = 0; currentIteration_ < maxIteration_; ++ currentIteration_) {
 

    // Ap = Ap_{j} = Dslash * p_{j} ----> outputBuffer_[1];
    dslashParam.fermionIn_MRHS  = p;
    dslashParam.fermionOut_MRHS = Ap;
    dslash_operator_->setParam(&dslashParam);
    dslash_operator_->apply();  // Ap = Dslash * p




    // norm <r, r0>
    outputDotArg.input1    = reinterpret_cast<Complex<OutputFloat>*>(r);
    outputDotArg.input2    = reinterpret_cast<Complex<OutputFloat>*>(r0);
    outputDotArg.resArr    = reinterpret_cast<Complex<OutputFloat>*>(r_r0_norm);
    outputDotArg.tmpBuffer = reinterpret_cast<Complex<OutputFloat>*>(reduceBuffer);
    outputDotArg.vec_len   = vol * single_complex_vec_len / 2;
    interior_operator_.output_dotc(outputDotArg); // norm <r, r0>  ----> r_r0_norm
    // norm <Ap, r0>
    outputDotArg.input1    = reinterpret_cast<Complex<OutputFloat>*>(Ap);
    outputDotArg.resArr    = reinterpret_cast<Complex<OutputFloat>*>(Ap_p_norm);
    outputDotArg.vec_len   = vol * single_complex_vec_len / 2;
    interior_operator_.output_dotc(outputDotArg); // norm <Ap, r0> ----> output_scala_array_[3]

    // alpha = <r, r0> / <Ap, r0>
    output_elementwise_div_arg.res     = alpha_array;
    output_elementwise_div_arg.x       = reinterpret_cast<Complex<OutputFloat>*>(r_r0_norm);
    output_elementwise_div_arg.y       = reinterpret_cast<Complex<OutputFloat>*>(Ap_p_norm);
    output_elementwise_div_arg.vec_len = mInput;
    interior_operator_.output_elementwise_div(output_elementwise_div_arg); // alpha = <r, r0> / <Ap, r0>

    // s_{j} = r_{j} - alpha_{j} * Ap_{j} .     ----> xsay 存放到outputBuffer_[3]
    output_xsay_arg.res = reinterpret_cast<Complex<OutputFloat>*>(s);
    output_xsay_arg.x   = reinterpret_cast<Complex<OutputFloat>*>(r);
    output_xsay_arg.a   = reinterpret_cast<Complex<OutputFloat>*>(alpha_array);
    output_xsay_arg.y   = reinterpret_cast<Complex<OutputFloat>*>(Ap);
    interior_operator_.output_xsay(output_xsay_arg); // s = r - alpha * Ap
  
    // As = As_{j} = Dslash * s_{j} ----> outputBuffer_[2]
    dslashParam.fermionIn_MRHS = s;
    dslashParam.fermionOut_MRHS = As;
    dslash_operator_->setParam(&dslashParam);
    dslash_operator_->apply();  // As = Dslash * s

    // omega = <As, s> / <As, As>
    // norm <As, s> -------> output_scala_array_[2]
    outputDotArg.input1 = reinterpret_cast<Complex<OutputFloat>*>(As);
    outputDotArg.input2 = reinterpret_cast<Complex<OutputFloat>*>(s);
    outputDotArg.resArr = reinterpret_cast<Complex<OutputFloat>*>(As_s_norm);
    outputDotArg.vec_len   = vol * single_complex_vec_len / 2;
    interior_operator_.output_dotc(outputDotArg);

    // <As, As>     -----> output_scala_array_[3]
    outputDotArg.input2  = reinterpret_cast<Complex<OutputFloat>*>(As);
    outputDotArg.resArr  = reinterpret_cast<Complex<OutputFloat>*>(As_As_norm);
    outputDotArg.vec_len = vol * single_complex_vec_len / 2;
    interior_operator_.output_dotc(outputDotArg);

    // omega = <As, s> / <As, As>
    output_elementwise_div_arg.res = omega_array;
    output_elementwise_div_arg.x   = As_s_norm;
    output_elementwise_div_arg.y   = As_As_norm; 
    output_elementwise_div_arg.vec_len = mInput;
    interior_operator_.output_elementwise_div(output_elementwise_div_arg);
    // x_new = x + alpha * p + omega * s
    output_axpbypcz_arg.res = reinterpret_cast<Complex<OutputFloat>*>(x_new);
    output_axpbypcz_arg.a   = reinterpret_cast<Complex<OutputFloat>*>(one_array);
    output_axpbypcz_arg.x   = reinterpret_cast<Complex<OutputFloat>*>(x_new);
    output_axpbypcz_arg.b   = reinterpret_cast<Complex<OutputFloat>*>(alpha_array);
    output_axpbypcz_arg.y   = reinterpret_cast<Complex<OutputFloat>*>(p);
    output_axpbypcz_arg.c   = reinterpret_cast<Complex<OutputFloat>*>(omega_array);
    output_axpbypcz_arg.z   = reinterpret_cast<Complex<OutputFloat>*>(s);
    output_axpbypcz_arg.single_vec_len = vol * single_complex_vec_len / 2;
    output_axpbypcz_arg.inc_idx = mInput;
    output_axpbypcz_arg.stream = stream1;
    interior_operator_.output_axpbypcz(output_axpbypcz_arg); // x_new = x + alpha * p + omega * s

    // r_new = s - omega * As
    output_xsay_arg.res = reinterpret_cast<Complex<OutputFloat>*>(r_new);
    output_xsay_arg.x   = reinterpret_cast<Complex<OutputFloat>*>(s);
    output_xsay_arg.a   = reinterpret_cast<Complex<OutputFloat>*>(omega_array);
    output_xsay_arg.y   = reinterpret_cast<Complex<OutputFloat>*>(As);
    interior_operator_.output_xsay(output_xsay_arg); // s = r - alpha * Ap

    // if converge ?
    { // converge return true
      // calculate norm of r_new and store them
      void* r_new_norm = output_scala_array_[3];
      output_norm_arg.input = reinterpret_cast<Complex<OutputFloat>*>(r_new);
      output_norm_arg.resArr = reinterpret_cast<OutputFloat*>(r_new_norm);
      interior_operator_.output_norm(output_norm_arg); // 计算norm，一次性保存到host端
      CHECK_CUDA(cudaMemcpy(norm_r_array.data(), 
                            r_new_norm,
                            sizeof(OutputFloat) * mInput, 
                            cudaMemcpyDeviceToHost,
                            stream1));
      CHECK_CUDA(cudaStreamSynchronize(stream1));
      bool is_converged = isConverged(norm_r_array, norm_b_array, maxPrec_);
      if (is_converged) {
        return true;
      }
    }
    // beta =  (alpha / omega)(<r_new, r0> / <r, r0>)
    // we now have <r, r0> in r_r0_norm
    // now calculate <r_new, r0> and store it in r_new_r0_norm
    outputDotArg.input1 = reinterpret_cast<Complex<OutputFloat>*>(r_new);
    outputDotArg.input2 = reinterpret_cast<Complex<OutputFloat>*>(r0);
    outputDotArg.resArr = reinterpret_cast<Complex<OutputFloat>*>(r_new_r0_norm);
    outputDotArg.vec_len = vol * single_complex_vec_len / 2;
    interior_operator_.output_dotc(outputDotArg); // <r_new, r0> ----> r_new_r0_norm
    // beta_temp = alpha * r_r0_norm / omega / r_new_r0_norm
    output_elementwise_mul_arg.res     = beta_array;
    output_elementwise_mul_arg.x       = reinterpret_cast<Complex<OutputFloat>*>(alpha_array);
    output_elementwise_mul_arg.y       = reinterpret_cast<Complex<OutputFloat>*>(r_r0_norm);
    output_elementwise_mul_arg.vec_len = mInput;
    interior_operator_.output_elementwise_mul(output_elementwise_mul_arg); // alpha * r_r0_norm ----> beta_array

    output_elementwise_div_arg.res     = beta_array;
    output_elementwise_div_arg.x       = reinterpret_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.y       = reinterpret_cast<Complex<OutputFloat>*>(omega_array);
    output_elementwise_div_arg.vec_len = mInput;
    interior_operator_.output_elementwise_div(output_elementwise_div_arg); // beta / omega ----> beta_array

    output_elementwise_div_arg.res     = beta_array;
    output_elementwise_div_arg.x       = reinterpret_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.y       = reinterpret_cast<Complex<OutputFloat>*>(r_new_r0_norm);
    output_elementwise_div_arg.vec_len = mInput;
    interior_operator_.output_elementwise_div(output_elementwise_div_arg); // beta / r_new_r0_norm ----> beta_array
    // p_new = r_new + beta * (p - omega * Ap)
    // first step: p_new = p - omega * Ap
    output_xsay_arg.res = reinterpret_cast<Complex<OutputFloat>*>(p_new);
    output_xsay_arg.x   = reinterpret_cast<Complex<OutputFloat>*>(p);
    output_xsay_arg.a   = reinterpret_cast<Complex<OutputFloat>*>(omega_array);
    output_xsay_arg.y   = reinterpret_cast<Complex<OutputFloat>*>(Ap);
    interior_operator_.output_xsay(output_xsay_arg); // s = r - alpha * Ap
    // second step: p_new = r_new + beta * p_new
    output_xpay_arg.res = reinterpret_cast<Complex<OutputFloat>*>(p_new);
    output_xpay_arg.x   = reinterpret_cast<Complex<OutputFloat>*>(r_new);
    output_xpay_arg.a   = reinterpret_cast<Complex<OutputFloat>*>(beta_array);
    output_xpay_arg.y   = reinterpret_cast<Complex<OutputFloat>*>(p_new);
    output_xpay_arg.single_vec_len = vol * single_complex_vec_len / 2;
    output_xpay_arg.inc_idx = mInput;
    output_xpay_arg.stream = stream1;
    interior_operator_.output_xpay(output_xpay_arg); // p_new = r_new + beta * p_new

    // r = r_new, p = p_new, x = x_new
    r_new = r;
    p_new = p;
    x_new = x;
  }

  return currentIteration_ <= maxIteration_ && isConverged(norm_r_array, norm_b_array, maxPrec_);
}

template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION InputPrecision,
          QCU_PRECISION IteratePrecision>
bool BiCGStab<OutputPrecision, InputPrecision, IteratePrecision>::solve_even() {
  const int Lx     = exterior_dslashParam_->lattDesc->X();
  const int Ly     = exterior_dslashParam_->lattDesc->Y();
  const int Lz     = exterior_dslashParam_->lattDesc->Z();
  const int Lt     = exterior_dslashParam_->lattDesc->T();
  const int mInput = exterior_dslashParam_->mInput;
  const int vol = Lx * Ly * Lz * Lt;
  const int single_complex_vec_len = exterior_dslashParam_->nColor * Ns;
  const int complex_vec_len =   exterior_dslashParam_->mInput 
                              * exterior_dslashParam_->nColor 
                              * Ns; // on single point
  // solve x_e
  // x_e = b_e + kappa D_{eo} x_{o}

  // now we get x_{o} and b_{e}
  void* x_e = outputBuffer_[9];
  void* x_o = outputBuffer_[10];
  void* b_e = new_b_output_prec_;
  void* kappa_array = output_scala_array_[0];
  // D_{eo} x_{o} ----> x_e
  DslashParam dslashParam {
    false,                                      // bool p_daggerFlag,
    OutputPrecision,                            // QCU_PRECISION p_precision,
    exterior_dslashParam_->nColor,              // int p_nColor, 
    exterior_dslashParam_->mInput,              // int p_mInput, 
    ODD_PARITY,                                 // int p_parity,
    OutputFloat(exterior_dslashParam_->kappa),  // double p_kappa, 
    x_o,                                        // void* p_fermionIn_MRHS
    x_e,                                        // void* p_fermionOut_MRHS,
    exterior_dslashParam_->gauge,               // void* p_gauge, 
    exterior_dslashParam_->lattDesc,            // const QcuLattDesc* p_lattDesc,
    exterior_dslashParam_->procDesc,            // const QcuProcDesc* p_procDesc,
    exterior_dslashParam_->stream1,             // cudaStream_t p_stream1 = NULL,
    exterior_dslashParam_->stream2              // cudaStream_t p_stream2 = NULL)
  }; 
  dslash_operator_->setParam(&dslashParam);
  dslash_operator_->apply();  // x_e = D_{eo} x_{o}

  // x_e = b_e + kappa D_{eo} x_{o}
  //     = b_e + kappa x_e
  using Output_xpayArgument = typename InteriorOperator::Output_xpayAruArgument;
  Output_xpayArgument output_xpay_arg {
      static_cast<Complex<OutputFloat>*>(x_e),          // Complex<_Float>* res,
      static_cast<Complex<OutputFloat>*>(b_e),          // Complex<_Float>* x,
      static_cast<Complex<OutputFloat>*>(kappa_array),  // Complex<_Float>* a,
      static_cast<Complex<OutputFloat>*>(x_e),          // Complex<_Float>* y,
      vol * single_complex_vec_len / 2,                 // int single_vec_len,
      mInput,                                           // int inc_idx,
      exterior_dslashParam_->stream1
  }; 
  interior_operator_.output_xpay(output_xpay_arg); // x_e = b_e + kappa x_e

  CHECK_CUDA(cudaStreamSynchronize(exterior_dslashParam_->stream1));
  return true;
}

template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION InputPrecision,
          QCU_PRECISION IteratePrecision>
bool BiCGStab<OutputPrecision, InputPrecision, IteratePrecision>::solve() {
  if (!bufferAllocated_) {
    if (!tempBufferAllocate()) {
      return false;
    }
  }

  if (!solve_odd()) {
    printf("solve odd failed, %d iterations\n", currentIteration_);
    return false;
  }
  if (!solve_even()) {
    printf("solve even failed, %d iterations\n", currentIteration_);
    return false;
  }

  printf("solve success, %d iterations\n", currentIteration_);
  const int Lx     = exterior_dslashParam_->lattDesc->X();
  const int Ly     = exterior_dslashParam_->lattDesc->Y();
  const int Lz     = exterior_dslashParam_->lattDesc->Z();
  const int Lt     = exterior_dslashParam_->lattDesc->T();
  const int mInput = exterior_dslashParam_->mInput;
  const int vol = Lx * Ly * Lz * Lt;
  const int complex_vec_len =   exterior_dslashParam_->mInput 
                              * exterior_dslashParam_->nColor 
                              * Ns; // on single point
  // copy x to outputBuffer
  // copy x_even to fermionOut_MRHS_even
  // copy x_odd to fermionOut_MRHS_odd
  void* fermionOut_MRHS_even = exterior_dslashParam_->fermionOut_MRHS;
  void* fermionIn_MRHS_even  = static_cast<Complex<OutputFloat>*>(exterior_dslashParam_->fermionIn_MRHS)
                                + vol * complex_vec_len / 2;
  return true;
}
}