#include <thrust/system/cuda/detail/par.h>

#include <type_traits>
#include <vector>

#include "data_format/qcu_data_format_shift.cuh"
#include "solver/bicgstab.cuh"

#define DEBUG
#ifdef DEBUG
template <typename _Float>
void checkNorm (void* global_mem, int round = 1) {
  if (round == 0) {
    _Float x;
    cudaMemcpy(&x, global_mem, sizeof(_Float), cudaMemcpyDeviceToHost);
    std::printf("norm = %lf\n", x);
  }
}
#endif

// out = in - a DoeDeo in
template <typename _Float>
static inline void fused_x_sub_Doe_Deo_x (void* output, void* input, void* temp, void* a,
                                          qcu::Dslash* dslash, qcu::DslashParam& param)
{
  const int Lx = param.lattDesc->X();
  const int Ly = param.lattDesc->Y();
  const int Lz = param.lattDesc->Z();
  const int Lt = param.lattDesc->T();
  const int vol = Lx * Ly * Lz * Lt;
  const int mInput = param.mInput;
  const int nColor = param.nColor;
  const int single_vec_len = Nd * nColor;

  cudaStream_t stream1 = param.stream1;
  cudaStream_t stream2 = param.stream2;
  // temp = Deo in
  param.fermionOut_MRHS = temp;
  param.fermionIn_MRHS = input;
  param.parity = EVEN_PARITY;
  dslash->setParam(&param);
  dslash->apply();
  CHECK_CUDA(cudaStreamSynchronize(stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream2));

  // out = Doe temp
  param.fermionOut_MRHS = output;
  param.fermionIn_MRHS = temp;
  param.parity = ODD_PARITY;
  dslash->setParam(&param);
  dslash->apply();
  CHECK_CUDA(cudaStreamSynchronize(stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream2));
  //
  // CHECK_CUDA(cudaDeviceSynchronize());
  // std::printf("=====<in> = ");
  // checkNorm<_Float>(input, 0);
  // std::printf("Doe in = ");
  // checkNorm<_Float>(temp, 0);
  // std::printf("Deo Doe in = ");
  // checkNorm<_Float>(output, 0);
  // CHECK_CUDA(cudaDeviceSynchronize());

  typename qcu::qcu_blas::Complex_xsay<_Float>::template Complex_xsayArgument
              xsay_argument {
                static_cast<Complex<_Float>*>(output),
                static_cast<Complex<_Float>*>(input),   // Complex<_Float>* x,
                static_cast<Complex<_Float>*>(a),       // Complex<_Float>* a,
                static_cast<Complex<_Float>*>(output),  // Complex<_Float>* y,
                single_vec_len * vol / 2,               // int single_vec_len,
                mInput,                                 // int inc_idx,
                stream1                                 // cudaStream_t stream = nullptr
              };
  qcu::qcu_blas::Complex_xsay<_Float> xsay_op;
  xsay_op(xsay_argument);
  CHECK_CUDA(cudaStreamSynchronize(stream1));
  //
  // CHECK_CUDA(cudaDeviceSynchronize());
  // std::printf("in - kappa * kappa Deo Doe in = ");
  // checkNorm<_Float>(output, 0);
  // CHECK_CUDA(cudaDeviceSynchronize());
}

namespace qcu::solver {
template <typename _Float, std::enable_if_t<std::is_same_v<_Float, float> ||
                                            std::is_same_v<_Float, double>>* = nullptr>
inline bool isConverged ( const std::vector<_Float>& norm_r_array, 
                          const std::vector<_Float>& norm_b_array,
                          _Float target_diff)
{
  // calculate the relative error
  _Float total_error = _Float(0.0);
  int size = norm_r_array.size();
  for (int i = 0; i < size; ++i) {
    total_error += (norm_r_array[i]) / norm_b_array[i];
    std::printf(" norm_r = %lf, norm_b = %lf\n", norm_r_array[i], norm_b_array[i]);
  }

  return (total_error / size) < target_diff;
}

// separator
template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION IteratePrecision>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve_odd() {
  std::vector<OutputFloat> norm_r_array  (param_.mInput, 1.0);
  std::vector<OutputFloat> norm_b_array  (param_.mInput, 1.0);  // 计算b的模长
  // diff_array = [r1, r2, r3, ...] / [b1, b2, b3, ...]
  const int Lx     = param_.lattDesc->X();
  const int Ly     = param_.lattDesc->Y();
  const int Lz     = param_.lattDesc->Z();
  const int Lt     = param_.lattDesc->T();
  const int mInput = param_.mInput;
  const int vol    = Lx * Ly * Lz * Lt;
  const int single_complex_vec_len = param_.nColor * Ns;
  const int complex_vec_len =   param_.mInput * single_complex_vec_len; // m-rhs on single point

  cudaStream_t stream1 = param_.stream1;
  cudaStream_t stream2 = param_.stream2;
  // prelogue
  // solve x_o
  // get new B
  reCalculate_b_even ();

  // calculate norm of b and store them
  using IterNormArgument   = typename InteriorOperator::template ComplexNorm<OutputFloat, IterateFloat>
                                                      ::template ComplexNormArgument;

  void* b = new_b_output_prec_;
  void* output_new_b_even_norm = output_scala_array_[2];

  // Ax存入outputBuffer_[1]，接下来计算r = b - Ax放入outputBuffer_[0]
  // r0 = b - Ax = b ----->r0，整个迭代过程不改变, 由于r0不变，初始化为b，因此直接指向b
  void* const r0 = outputBuffer_[0];
  void*       r  = outputBuffer_[10]; // 初始时 r0 == r1   因为要求<r0, r> 不为0
  void*       p  = outputBuffer_[11]; // p = r0，从此outputBuffer[1]不变，不能再使用

  void* Ap           = outputBuffer_[1];  // 每轮要留住 outputBuffer_[1]作为 Ap
  void* As           = outputBuffer_[2];  // 每轮要留住 outputBuffer_[2]作为 As
  void* s            = outputBuffer_[3];
  void* r_new        = outputBuffer_[4];
  void* reduceBuffer = outputBuffer_[5];  // use this tempBuffer to reduce
  void* x_new        = outputBuffer_[6];
  void* p_new        = outputBuffer_[7];
  // void* x            = outputBuffer_[8];
  void* x_o          = static_cast<Complex<OutputFloat>*>(result_x_output_prec_) + vol / 2 * complex_vec_len;
  void* temp_buffer  = outputBuffer_[9];

  void* kappa_square_array = output_scala_array_[5];
  void* one_array          = output_scala_array_[1];
  void* r_r0_norm          = output_scala_array_[2]; // 要用到最后
  void* Ap_dot_r0          = output_scala_array_[3];
  void* r_new_r0_norm      = output_scala_array_[4];
  void* As_s_norm          = output_scala_array_[8];
  void* As_As_norm         = output_scala_array_[6];
  void* r_new_norm         = output_scala_array_[7];
  using OutputNormArgument = typename InteriorOperator::template ComplexNorm<OutputFloat, OutputFloat>
                                                      ::template ComplexNormArgument;
  OutputNormArgument output_norm_arg {
    vol * single_complex_vec_len / 2,                      // single_vec_len;
    mInput,                                                // stride;
    static_cast<OutputFloat*>(temp_buffer),                // tmpBuffer
    static_cast<Complex<OutputFloat>*>(b),                 // input
    static_cast<OutputFloat*>(output_new_b_even_norm),     // resArr
    stream1,
    cublasHandle_
  };

  interior_operator_.output_norm(output_norm_arg);
  // 计算norm，一次性保存到host端
  CHECK_CUDA(cudaMemcpyAsync( norm_b_array.data(), output_new_b_even_norm,
                              sizeof(OutputFloat) * mInput,
                              cudaMemcpyDeviceToHost, stream1));
  // store norm of b in norm_b_array (host)
  CHECK_CUDA(cudaStreamSynchronize(stream1));

  // R = b - A * x = b - Dslash * x, x可以初始化为0
  DslashParam dslashParam {
    false,                      // daggerFlag,
    OutputPrecision,            // dslash precision
    param_.nColor,              // nColor,
    param_.mInput,              // mInput,
    EVEN_PARITY,                // parity,
    OutputFloat(param_.kappa),  // kappa,
    b,    // fermionIn_MRHS
    outputBuffer_[1],           // fermionOut_MRHS,
    param_.gauge,               // gauge,
    param_.lattDesc,            // lattDesc,
    param_.procDesc,            // procDesc,
    stream1,                    // stream1 = NULL,
    stream2                     // stream2 = NULL
  };

  using Output_xsayArgument = typename InteriorOperator::template Complex_xsay<OutputFloat>
                                                       ::template Complex_xsayArgument;

  Output_xsayArgument output_xsay_arg { nullptr, nullptr, nullptr, nullptr,
      vol * single_complex_vec_len / 2,                   // int single_vec_len,
      mInput,                                             // int inc_idx,
      stream1
  };

  // InnerProduct Param
  using OutputDotcArgument = typename InteriorOperator::template ComplexDotc<OutputFloat, OutputFloat>
                                                      ::template DotcArgument;
  OutputDotcArgument outputDotArg {
    vol * single_complex_vec_len / 2, // single_vec_len;
    mInput,                           // stride;
    nullptr,                          // tmpBuffer;
    nullptr,                          // input1;
    nullptr,                          // input2;
    nullptr,                          // resArr;
    stream1,                          // stream;
    cublasHandle_,                    // handle;
  };
  // ElementwiseDiv Param
  using OutputElementwiseDivArgument = typename InteriorOperator::template ElementwiseDiv<Complex<OutputFloat>>
                                                                ::template ElementwiseDivArgument;
  OutputElementwiseDivArgument output_elementwise_div_arg { nullptr, nullptr, nullptr, mInput /* vec_len */ };

  // ElementwiseMul Param
  using OutputElementwiseMulArgument = typename InteriorOperator::template ElementwiseMul<Complex<OutputFloat>>
                                                                ::template ElementwiseMulArgument;
  OutputElementwiseMulArgument output_elementwise_mul_arg { nullptr, nullptr, nullptr, mInput /* vec_len */ };

  using OutputAxpbypczArgument = typename InteriorOperator::template Complex_axpbypcz<OutputFloat>
                                                          ::template Complex_axpbypczArgument;
  OutputAxpbypczArgument output_axpbypcz_arg {nullptr, nullptr, nullptr,nullptr, nullptr, nullptr, nullptr,
    vol * single_complex_vec_len / 2 /* single_vec_len */, mInput /* inc_idx */, stream1 /* stream*/ };

  using Output_xpayArgument = typename InteriorOperator::template Complex_xpay<OutputFloat>
                                                        ::template Complex_xpayArgument;
  Output_xpayArgument output_xpay_arg {
      nullptr, nullptr, nullptr, nullptr,
      vol * single_complex_vec_len / 2 /*single_vec_len*/,
      mInput /*inc_idx*/, stream1 };

  // prelogue
  // x = 0
  CHECK_CUDA(cudaMemsetAsync(x_new, vol / 2 * complex_vec_len, 0, stream1));
  // r = p = r0 = b - Ax = b
  CHECK_CUDA(cudaMemcpyAsync(r, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
  CHECK_CUDA(cudaMemcpyAsync(r0, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
  CHECK_CUDA(cudaMemcpyAsync(p, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream1));

  // output_norm_arg.input = static_cast<Complex<OutputFloat>*>(p);
  // output_norm_arg.resArr = static_cast<OutputFloat*>(output_new_b_even_norm);
  // CHECK_CUDA(cudaDeviceSynchronize());

  // begin iteration
  // 开始迭代，达到最大迭代次数不收敛则返回false
  for (currentIteration_ = 0; currentIteration_ < maxIteration_; ++ currentIteration_) {

    // Ap = Ap_{j} = Doe Deo * p_{j} ----> outputBuffer_[1];
    fused_x_sub_Doe_Deo_x<OutputFloat>(Ap, p, temp_buffer, kappa_square_array, dslash_operator_, dslashParam);
    cudaStreamSynchronize(dslashParam.stream1);
    cudaStreamSynchronize(dslashParam.stream2);

    // CHECK_CUDA(cudaDeviceSynchronize());
    // std::printf("=====iteration = %d, <p> = ", currentIteration_);
    // checkNorm<OutputFloat>(p, 0);
    //
    // CHECK_CUDA(cudaDeviceSynchronize());
    // std::printf("<kappa * kappa> = ");
    // checkNorm<OutputFloat>(kappa_square_array, 0);

    // norm <r, r0>
    outputDotArg.input1    = static_cast<Complex<OutputFloat>*>(r);
    outputDotArg.input2    = static_cast<Complex<OutputFloat>*>(r0);
    outputDotArg.resArr    = static_cast<Complex<OutputFloat>*>(r_r0_norm);
    outputDotArg.tmpBuffer = static_cast<Complex<OutputFloat>*>(reduceBuffer);
    interior_operator_.output_dotc(outputDotArg); // norm <r, r0>  ----> r_r0_norm

    // norm <Ap, r0>
    outputDotArg.input1    = static_cast<Complex<OutputFloat>*>(Ap);
    outputDotArg.resArr    = static_cast<Complex<OutputFloat>*>(Ap_dot_r0);
    interior_operator_.output_dotc(outputDotArg); // norm <Ap, r0> ----> output_scala_array_[3]




    // alpha = <r, r0> / <Ap, r0>
    output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(alpha_array);
    output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(r_r0_norm);
    output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(Ap_dot_r0);
    interior_operator_.output_elementwise_div(output_elementwise_div_arg); // alpha = <r, r0> / <Ap, r0>
    CHECK_CUDA(cudaStreamSynchronize(output_elementwise_div_arg.stream));

    // s_{j} = r_{j} - alpha_{j} * Ap_{j} .     ----> xsay 存放到outputBuffer_[3]
    output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(s);
    output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(r);
    output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(alpha_array);
    output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(Ap);
    interior_operator_.output_xsay(output_xsay_arg); // s = r - alpha * Ap

    // As = As_{j} = Doe Deo * s_{j}
    fused_x_sub_Doe_Deo_x<OutputFloat>(As, s, temp_buffer, kappa_square_array, dslash_operator_, dslashParam);

    // omega = <As, s> / <As, As>
    // norm <As, s> -------> output_scala_array_[2]
    outputDotArg.input1 = static_cast<Complex<OutputFloat>*>(As);
    outputDotArg.input2 = static_cast<Complex<OutputFloat>*>(s);
    outputDotArg.resArr = static_cast<Complex<OutputFloat>*>(As_s_norm);
    interior_operator_.output_dotc(outputDotArg);

    // <As, As>     -----> output_scala_array_[3]
    outputDotArg.input2  = static_cast<Complex<OutputFloat>*>(As);
    outputDotArg.resArr  = static_cast<Complex<OutputFloat>*>(As_As_norm);
    interior_operator_.output_dotc(outputDotArg);

    // omega = <As, s> / <As, As>
    output_elementwise_div_arg.res = static_cast<Complex<OutputFloat>*>(omega_array);
    output_elementwise_div_arg.x   = static_cast<Complex<OutputFloat>*>(As_s_norm);
    output_elementwise_div_arg.y   = static_cast<Complex<OutputFloat>*>(As_As_norm);
    interior_operator_.output_elementwise_div(output_elementwise_div_arg);

    // x_new = x + alpha * p + omega * s
    output_axpbypcz_arg.res = static_cast<Complex<OutputFloat>*>(x_new);
    output_axpbypcz_arg.a   = static_cast<Complex<OutputFloat>*>(one_array);
    output_axpbypcz_arg.x   = static_cast<Complex<OutputFloat>*>(x_new);
    output_axpbypcz_arg.b   = static_cast<Complex<OutputFloat>*>(alpha_array);
    output_axpbypcz_arg.y   = static_cast<Complex<OutputFloat>*>(p);
    output_axpbypcz_arg.c   = static_cast<Complex<OutputFloat>*>(omega_array);
    output_axpbypcz_arg.z   = static_cast<Complex<OutputFloat>*>(s);
    output_axpbypcz_arg.stream = stream1;
    interior_operator_.output_axpbypcz(output_axpbypcz_arg); // x_new = x + alpha * p + omega * s

    // r_new = s - omega * As
    output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(r_new);
    output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(s);
    output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(omega_array);
    output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(As);
    interior_operator_.output_xsay(output_xsay_arg); // r_new = s - omega * Ap

    // if converge ?
    { // converge return true
      // calculate norm of r_new and store them
      // void* r_new_norm = output_scala_array_[3];
      output_norm_arg.input  = static_cast<Complex<OutputFloat>*>(r_new);
      output_norm_arg.resArr = static_cast<OutputFloat*>(r_new_norm);
      interior_operator_.output_norm(output_norm_arg); // 计算norm，一次性保存到host端
      CHECK_CUDA(cudaMemcpyAsync(norm_r_array.data(), r_new_norm,
                            sizeof(OutputFloat) * mInput,
                            cudaMemcpyDeviceToHost, stream1));
      CHECK_CUDA(cudaStreamSynchronize(stream1));

      std::printf("DEBUG, currentIteration = %d", currentIteration_);
      if (bool is_converged = isConverged<OutputFloat>(norm_r_array, norm_b_array, maxPrec_)) {
        CHECK_CUDA(cudaMemcpyAsync(x_o, x_new, sizeof(OutputFloat) * vol / 2 * complex_vec_len * 2,
                              cudaMemcpyDeviceToDevice, stream1)); // res_x = x_new = x_{j + 1}
        CHECK_CUDA(cudaStreamSynchronize(stream1));

        // CHECK_CUDA(cudaDeviceSynchronize());
        // std::printf("=====iteration = %d, <res> = ", currentIteration_);
        // checkNorm<OutputFloat>(x_new, 0);
        return true;
      }
    }
    // beta =  (alpha / omega)(<r_new, r0> / <r, r0>)
    // we now have <r, r0> in r_r0_norm
    // now calculate <r_new, r0> and store it in r_new_r0_norm
    outputDotArg.input1 = static_cast<Complex<OutputFloat>*>(r_new);
    outputDotArg.input2 = static_cast<Complex<OutputFloat>*>(r0);
    outputDotArg.resArr = static_cast<Complex<OutputFloat>*>(r_new_r0_norm);
    interior_operator_.output_dotc(outputDotArg); // <r_new, r0> ----> r_new_r0_norm

    // beta_temp = alpha * r_new_r0_norm / omega / r_r0_norm
    // first step : beta = alpha * <r_new, r0>
    output_elementwise_mul_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_mul_arg.x       = static_cast<Complex<OutputFloat>*>(alpha_array);
    output_elementwise_mul_arg.y       = static_cast<Complex<OutputFloat>*>(r_new_r0_norm);
    interior_operator_.output_elementwise_mul(output_elementwise_mul_arg); // alpha * r_new_r0_prod ----> beta_array

    // second step : beta = beta / omega = alpha * r_new_r0_norm / omega
    output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(omega_array);
    interior_operator_.output_elementwise_div(output_elementwise_div_arg); // beta / omega ----> beta_array

    // third step : beta = beta / r_r0_norm
    output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(r_r0_norm);
    interior_operator_.output_elementwise_div(output_elementwise_div_arg); // beta / r_r0_norm ----> beta_array

    // CHECK_CUDA(cudaDeviceSynchronize());
    // std::printf("<beta> = ");
    // checkNorm<OutputFloat>(beta_array, 0);

    // p_new = r_new + beta * (p - omega * Ap)
    // first step: p_new = p - omega * Ap
    output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(p_new);
    output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(p);
    output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(omega_array);
    output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(Ap);
    interior_operator_.output_xsay(output_xsay_arg); // s = p - omega * Ap
    //
    // CHECK_CUDA(cudaDeviceSynchronize());
    // std::printf("<p - omega * Ap> = ");
    // checkNorm<OutputFloat>(p_new, 0);

    // second step: p_new = r_new + beta * p_new
    output_xpay_arg.res = static_cast<Complex<OutputFloat>*>(p_new);
    output_xpay_arg.x   = static_cast<Complex<OutputFloat>*>(r_new);
    output_xpay_arg.a   = static_cast<Complex<OutputFloat>*>(beta_array);
    output_xpay_arg.y   = static_cast<Complex<OutputFloat>*>(p_new);
    output_xpay_arg.stream = stream1;
    interior_operator_.output_xpay(output_xpay_arg); // p_new = r_new + beta * p_new
    //
    // CHECK_CUDA(cudaDeviceSynchronize());
    // std::printf("iteration = %d, <p_new> = ", currentIteration_);
    // checkNorm<OutputFloat>(p_new, 0);

    std::swap(r, r_new);  // r_new = r
    std::swap(p, p_new);  // p_new = p
    // std::swap(x, x_new);  // x_new = x
  }

  return currentIteration_ < maxIteration_ && isConverged(norm_r_array, norm_b_array, maxPrec_);
}

template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION IteratePrecision>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve_even() {
  const int Lx     = param_.lattDesc->X();
  const int Ly     = param_.lattDesc->Y();
  const int Lz     = param_.lattDesc->Z();
  const int Lt     = param_.lattDesc->T();
  const int mInput = param_.mInput;
  const int vol = Lx * Ly * Lz * Lt;
  const int single_complex_vec_len = param_.nColor * Ns;
  const int mrhs_complex_vec_len = param_.mInput * single_complex_vec_len;
  // solve x_e
  // x_e = b_e + kappa D_{eo} x_{o}
  // now we get x_{o} and b_{e}
  void* x_e = result_x_output_prec_;
  void* x_o = static_cast<Complex<OutputFloat>*>(x_e) + vol / 2 * mrhs_complex_vec_len;
  void* b_e_origin = param_.input_b_mrhs;;
  void* kappa_array = output_scala_array_[0];

  // D_{eo} x_{o} ----> x_e
  DslashParam dslashParam {
    false,                      // bool p_daggerFlag,
    OutputPrecision,            // QCU_PRECISION p_precision,
    param_.nColor,              // int p_nColor,
    param_.mInput,              // int p_mInput,
    EVEN_PARITY,                // int p_parity,
    OutputFloat(param_.kappa),  // double p_kappa,
    x_o,                        // void* p_fermionIn_MRHS
    x_e,                        // void* p_fermionOut_MRHS,
    param_.gauge,               // void* p_gauge,
    param_.lattDesc,            // const QcuLattDesc* p_lattDesc,
    param_.procDesc,            // const QcuProcDesc* p_procDesc,
    param_.stream1,             // cudaStream_t p_stream1 = NULL,
    param_.stream2              // cudaStream_t p_stream2 = NULL)
  }; 
  dslash_operator_->setParam(&dslashParam);
  dslash_operator_->apply();  // x_e = D_{eo} x_{o}

  CHECK_CUDA(cudaDeviceSynchronize());
  std::printf("x_o = ");
  checkNorm<OutputFloat>(x_o, 0);
  std::printf("Deo x_o = ");
  checkNorm<OutputFloat>(x_e, 0);

  CHECK_CUDA(cudaDeviceSynchronize());
  // x_e = b_e + kappa D_{eo} x_{o}
  //     = b_e + kappa x_e
  // using Output_xpayArgument = typename InteriorOperator::Output_xpayAruArgument;
  using Output_xpayArgument = typename InteriorOperator::template Complex_xpay<OutputFloat>
                                                        ::template Complex_xpayArgument;
  Output_xpayArgument output_xpay_arg {
      static_cast<Complex<OutputFloat>*>(x_e),          // Complex<_Float>* res,
      static_cast<Complex<OutputFloat>*>(b_e_origin),          // Complex<_Float>* x,
      static_cast<Complex<OutputFloat>*>(kappa_array),  // Complex<_Float>* a,
      static_cast<Complex<OutputFloat>*>(x_e),          // Complex<_Float>* y,
      vol / 2 * single_complex_vec_len,                 // int single_vec_len,
      mInput,                                           // int inc_idx,
      param_.stream1
  }; 
  interior_operator_.output_xpay(output_xpay_arg); // x_e = b_e + kappa x_e
  CHECK_CUDA(cudaStreamSynchronize(param_.stream1));
  return true;
}

template <QCU_PRECISION OutputPrecision,
          QCU_PRECISION IteratePrecision>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve() {
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
  const int Lx     = param_.lattDesc->X();
  const int Ly     = param_.lattDesc->Y();
  const int Lz     = param_.lattDesc->Z();
  const int Lt     = param_.lattDesc->T();
  const int vol = Lx * Ly * Lz * Lt;
  const int mrhs_vec_len = param_.mInput * param_.nColor * Ns; // on single point
  const cudaStream_t cuda_stream = param_.stream1;
  // copy x to outputBuffer
  copyComplexVector_interface(param_.output_x_mrhs, OutputPrecision,
                              result_x_output_prec_, OutputPrecision,
                              vol * mrhs_vec_len, cuda_stream);
  // CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
  CHECK_CUDA(cudaDeviceSynchronize());
  return true;
}

// donnot use HALF to be the output precision
template class BiCGStabImpl<QCU_DOUBLE_PRECISION, QCU_DOUBLE_PRECISION>;
template class BiCGStabImpl<QCU_DOUBLE_PRECISION, QCU_SINGLE_PRECISION>;
template class BiCGStabImpl<QCU_DOUBLE_PRECISION, QCU_HALF_PRECISION>;
template class BiCGStabImpl<QCU_SINGLE_PRECISION, QCU_DOUBLE_PRECISION>;
template class BiCGStabImpl<QCU_SINGLE_PRECISION, QCU_SINGLE_PRECISION>;
template class BiCGStabImpl<QCU_SINGLE_PRECISION, QCU_HALF_PRECISION>;
}