#include <qcu_config/qcu_config.h>
#include <thrust/system/cuda/detail/par.h>

#include <type_traits>
#include <vector>

#include "check_error/check_cuda.cuh"
#include "data_format/qcu_data_format_shift.cuh"
#include "solver/bicgstab.cuh"

// #define DEBUG
#ifdef DEBUG
template <typename _Float>
void checkNorm (void* global_mem, int round = 1) {
    if (round == 0) {
        _Float x;
        cudaMemcpy(&x, global_mem, sizeof(_Float), cudaMemcpyDeviceToHost);
        std::printf("norm = %lf\n", x);
    }
}
#else
template <typename _Float>
void checkNorm (void* global_mem, int round = 1) {}
#endif

// out = in - a DoeDeo in
template <typename _Float>
static inline void fused_x_sub_Doe_Deo_x (
    void* output, void* input, void* temp, void* a,
    std::shared_ptr<qcu::Dslash> dslash,
    std::shared_ptr<qcu::DslashParam> param)
{
    const int vol = qcu::config::lattice_volume_local();
    const int m_input = param->m_input;
    const int n_color = param->n_color;
    const int single_vec_len = Nd * n_color;

    cudaStream_t stream1 = param->stream1;
    cudaStream_t stream2 = param->stream2;
    // temp = Deo in
    param->fermion_out_MRHS = temp;
    param->fermion_in_MRHS = input;
    param->parity = EVEN_PARITY;
    dslash->apply(param);
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    // out = Doe temp
    param->fermion_out_MRHS = output;
    param->fermion_in_MRHS = temp;
    param->parity = ODD_PARITY;

    dslash->apply(param);
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    typename qcu::qcu_blas::Complex_xsay<_Float>::template Complex_xsayArgument
        xsay_argument {
            static_cast<Complex<_Float>*>(output),
            static_cast<Complex<_Float>*>(input),   // Complex<_Float>* x,
            static_cast<Complex<_Float>*>(a),       // Complex<_Float>* a,
            static_cast<Complex<_Float>*>(output),  // Complex<_Float>* y,
            single_vec_len * vol / 2,               // int single_vec_len,
            m_input,                                 // int inc_idx,
            stream1                                 // cudaStream_t stream = nullptr
        };
    qcu::qcu_blas::Complex_xsay<_Float> xsay_op;
    xsay_op(xsay_argument);
    CHECK_CUDA(cudaStreamSynchronize(stream1));
}

namespace qcu::solver {
template <
    typename _Float,
    std::enable_if_t<std::is_same_v<_Float, float> || std::is_same_v<_Float, double>>* = nullptr
>
inline bool isConverged ( const std::vector<_Float>& norm_r_array, 
    const std::vector<_Float>& norm_b_array, _Float target_diff, bool log = false)
{
    // calculate the relative error
    const int size = norm_r_array.size();
    for (int i = 0; i < size; ++i) {
        if (norm_r_array[i] / norm_b_array[i] > target_diff) {
            return false;
        }
    }
    return true;
}

template <typename _Float,
    std::enable_if_t<std::is_same_v<_Float, float> || std::is_same_v<_Float, double>>* = nullptr
>
inline bool isConverged_policy2 (const _Float norm_r, const _Float norm_b, _Float target_diff) {
    return norm_r / norm_b <= target_diff;
}

// separator
template <
    QcuPrecision OutputPrecision,
    QcuPrecision IteratePrecision
>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve_odd_policy1() {
    std::cout << "POLICY1 BICGStab: separated residual" << std::endl;
    std::vector<OutputFloat> norm_r_array  (param_.mInput, 1.0);
    std::vector<OutputFloat> norm_b_array  (param_.mInput, 1.0);  // 计算b的模长
    // diff_array = [r1, r2, r3, ...] / [b1, b2, b3, ...]
    const int mInput = param_.mInput;
    const int vol = param_.lattDesc->lattice_volume();
    const int single_complex_vec_len = param_.nColor * Ns;
    const int complex_vec_len =   param_.mInput * single_complex_vec_len; // m-rhs on single point

    cudaStream_t stream1 = param_.stream1;
    cudaStream_t stream2 = param_.stream2;
    // prelogue
    // solve x_o
    // get new B
    reCalculate_b_even ();

    // calculate norm of b and store them
    using IterNormArgument   = typename InteriorOperator::template ComplexNorm<OutputFloat, IterateFloat>::template ComplexNormArgument;

    void* b = new_b_output_prec_;
    void* output_new_b_even_norm = output_scala_array_[2];

  // Ax存入outputBuffer_[1]，接下来计算r = b - Ax放入outputBuffer_[0]
  // r0 = b - Ax = b ----->r0，整个迭代过程不改变, 由于r0不变，初始化为b，因此直接指向b
    void* const r0 = outputBuffer_[0];
    void*       rj = outputBuffer_[10]; // 初始时 r0 == rj   因为要求<r0, rj> 不为0
    void*       pj = outputBuffer_[11]; // pj = r0，从此outputBuffer[1]不变，不能再使用

    void* vj           = outputBuffer_[1];  // 每轮要留住 outputBuffer_[1]作为 vj = Ap
    void* t            = outputBuffer_[2];  // 每轮要留住 outputBuffer_[2]作为 t = As
    void* sj           = outputBuffer_[3];
    void* r_new        = outputBuffer_[4];
    void* reduceBuffer = outputBuffer_[5];  // use this tempBuffer to reduce
    void* x_new        = outputBuffer_[6];
    void* p_new        = outputBuffer_[7];
    // void* x            = outputBuffer_[8];
    void* x_o          = static_cast<Complex<OutputFloat>*>(result_x_output_prec_) + vol / 2 * complex_vec_len;
    void* temp_buffer  = outputBuffer_[9];

    void* kappa_square_array = output_scala_array_[5];
    void* one_array          = output_scala_array_[1];
    void* rho_j              = output_scala_array_[2]; // 要用到最后 rho_i = <r0, ri>
    void* r0_dot_vj          = output_scala_array_[3]; // r0_dot_vj = <r0, vj> = <r0, A pj>
    void* rho_new            = output_scala_array_[4]; // rho_new = <r0, r_{j+1}>
    void* t_dot_sj           = output_scala_array_[8]; // t_dot_sj = <As, sj>
    void* t_dot_t            = output_scala_array_[6]; // t_dot_t = <As, As>
    void* r_new_norm         = output_scala_array_[7];
    using OutputNormArgument =
        typename InteriorOperator::template ComplexNorm<OutputFloat, OutputFloat>::template ComplexNormArgument;

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
    std::shared_ptr<DslashParam> dslashParam = std::make_shared<DslashParam>(
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
    );

    using Output_xsayArgument =
        typename InteriorOperator::template Complex_xsay<OutputFloat>::template Complex_xsayArgument;

    Output_xsayArgument output_xsay_arg { nullptr, nullptr, nullptr, nullptr,
        vol * single_complex_vec_len / 2,                   // int single_vec_len,
        mInput,                                             // int inc_idx,
        stream1
    };

    // InnerProduct Param
    using OutputDotcArgument =
        typename InteriorOperator::template ComplexDotc<OutputFloat, OutputFloat>::template DotcArgument;

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
    using OutputElementwiseDivArgument =
        typename InteriorOperator::template ElementwiseDiv<Complex<OutputFloat>>::template ElementwiseDivArgument;

    OutputElementwiseDivArgument output_elementwise_div_arg { nullptr, nullptr, nullptr, mInput /* vec_len */ };

    // ElementwiseMul Param
    using OutputElementwiseMulArgument =
        typename InteriorOperator::template ElementwiseMul<Complex<OutputFloat>>::template ElementwiseMulArgument;
    OutputElementwiseMulArgument output_elementwise_mul_arg { nullptr, nullptr, nullptr, mInput /* vec_len */ };

    using OutputAxpbypczArgument =
        typename InteriorOperator::template Complex_axpbypcz<OutputFloat>::template Complex_axpbypczArgument;

    OutputAxpbypczArgument output_axpbypcz_arg {
        nullptr, nullptr, nullptr,nullptr, nullptr, nullptr, nullptr,
        vol * single_complex_vec_len / 2 /* single_vec_len */,
        mInput /* inc_idx */,
        stream1 /* stream*/
    };

    using Output_xpayArgument =
        typename InteriorOperator::template Complex_xpay<OutputFloat>::template Complex_xpayArgument;

    Output_xpayArgument output_xpay_arg {
        nullptr, nullptr, nullptr, nullptr,
        vol * single_complex_vec_len / 2 /*single_vec_len*/,
        mInput /*inc_idx*/, stream1
    };

    // prelogue
    // x = 0
    CHECK_CUDA(cudaMemsetAsync(x_new, vol / 2 * complex_vec_len, 0, stream1));
    // rj = pj = r0 = b - Ax = b
    CHECK_CUDA(cudaMemcpyAsync(rj, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
    CHECK_CUDA(cudaMemcpyAsync(r0, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
    CHECK_CUDA(cudaMemcpyAsync(pj, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream1));

    // begin iteration
    // 开始迭代，达到最大迭代次数不收敛则返回false
    for (currentIteration_ = 0; currentIteration_ < maxIteration_; ++ currentIteration_) {
        // rho_j = <r0, r_j>
        outputDotArg.input1    = static_cast<Complex<OutputFloat>*>(r0);
        outputDotArg.input2    = static_cast<Complex<OutputFloat>*>(rj);
        outputDotArg.resArr    = static_cast<Complex<OutputFloat>*>(rho_j);
        outputDotArg.tmpBuffer = static_cast<Complex<OutputFloat>*>(reduceBuffer);
        interior_operator_.output_dotc(outputDotArg);

        // vj = Ap = Ap_{j} = Doe Deo * p_{j} ----> outputBuffer_[1];
        fused_x_sub_Doe_Deo_x<OutputFloat>(vj, pj, temp_buffer, kappa_square_array, dslash_operator_, dslashParam);
        cudaStreamSynchronize(dslashParam->stream1);
        cudaStreamSynchronize(dslashParam->stream2);

        // r0_dot_vj = <r0, vj> = <r0, Ap_j>
        // , norm <r0, Ap>
        outputDotArg.input2    = static_cast<Complex<OutputFloat>*>(vj);
        outputDotArg.resArr    = static_cast<Complex<OutputFloat>*>(r0_dot_vj);
        interior_operator_.output_dotc(outputDotArg);

        // alpha = <r0, r_i> / <r0, A pj> = rho_i / r0_dot_vj
        output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(alpha_array);
        output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(rho_j);
        output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(r0_dot_vj);
        interior_operator_.output_elementwise_div(output_elementwise_div_arg);
        CHECK_CUDA(cudaStreamSynchronize(output_elementwise_div_arg.stream));

        // sj = rj - alpha_{j} * vj = rj - alpha_{j} * A pj
        output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(sj);
        output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(rj);
        output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(alpha_array);
        output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(vj);
        interior_operator_.output_xsay(output_xsay_arg);

        // t = A sj = Doe Deo * sj
        fused_x_sub_Doe_Deo_x<OutputFloat>(t, sj, temp_buffer, kappa_square_array, dslash_operator_, dslashParam);

        // omega = <As, s> / <As, As> = t_dot_sj / t_dot_t
        // step1:  t_dot_sj = <As, s> = <t, sj>
        outputDotArg.input1 = static_cast<Complex<OutputFloat>*>(t);
        outputDotArg.input2 = static_cast<Complex<OutputFloat>*>(sj);
        outputDotArg.resArr = static_cast<Complex<OutputFloat>*>(t_dot_sj);
        interior_operator_.output_dotc(outputDotArg);
        // step2: <As, As>     -----> output_scala_array_[3]
        outputDotArg.input2  = static_cast<Complex<OutputFloat>*>(t);
        outputDotArg.resArr  = static_cast<Complex<OutputFloat>*>(t_dot_t);
        interior_operator_.output_dotc(outputDotArg);
        // step3: omega = <As, s> / <As, As>
        output_elementwise_div_arg.res = static_cast<Complex<OutputFloat>*>(omega_array);
        output_elementwise_div_arg.x   = static_cast<Complex<OutputFloat>*>(t_dot_sj);
        output_elementwise_div_arg.y   = static_cast<Complex<OutputFloat>*>(t_dot_t);
        interior_operator_.output_elementwise_div(output_elementwise_div_arg);

        // x_new = x + alpha * pj + omega * sj
        output_axpbypcz_arg.res = static_cast<Complex<OutputFloat>*>(x_new);
        output_axpbypcz_arg.a   = static_cast<Complex<OutputFloat>*>(one_array);
        output_axpbypcz_arg.x   = static_cast<Complex<OutputFloat>*>(x_new);
        output_axpbypcz_arg.b   = static_cast<Complex<OutputFloat>*>(alpha_array);
        output_axpbypcz_arg.y   = static_cast<Complex<OutputFloat>*>(pj);
        output_axpbypcz_arg.c   = static_cast<Complex<OutputFloat>*>(omega_array);
        output_axpbypcz_arg.z   = static_cast<Complex<OutputFloat>*>(sj);
        output_axpbypcz_arg.stream = stream1;
        interior_operator_.output_axpbypcz(output_axpbypcz_arg); // x_new = x + alpha * pj + omega * sj

        // r_new = s - omega * As = s - omega t
        output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(r_new);
        output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(sj);
        output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(omega_array);
        output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(t);
        interior_operator_.output_xsay(output_xsay_arg);

        // if converge ?
        {   // converge return true
            // calculate norm of r_new and store them
            // void* r_new_norm = output_scala_array_[3];
            output_norm_arg.input  = static_cast<Complex<OutputFloat>*>(r_new);
            output_norm_arg.resArr = static_cast<OutputFloat*>(r_new_norm);
            interior_operator_.output_norm(output_norm_arg); // 计算norm，一次性保存到host端
            CHECK_CUDA(cudaMemcpyAsync(norm_r_array.data(), r_new_norm,
                                sizeof(OutputFloat) * mInput,
                                cudaMemcpyDeviceToHost, stream1));
            CHECK_CUDA(cudaStreamSynchronize(stream1));

            if (bool is_converged = isConverged<OutputFloat>(norm_r_array, norm_b_array, maxPrec_)) {
                CHECK_CUDA(cudaMemcpyAsync(x_o, x_new, sizeof(OutputFloat) * vol / 2 * complex_vec_len * 2,
                                cudaMemcpyDeviceToDevice, stream1)); // res_x = x_new = x_{j + 1}
                CHECK_CUDA(cudaStreamSynchronize(stream1));
                return true;
            }
        }
        // beta =  (alpha / omega)(<r0, r_new> / <r0, rj>) = (alpha / omega) (rho_new / rho_i)
        // we now have <r, r0> in rho_i
        // now calculate <r0, r_new> and store it in rho_new
        outputDotArg.input1 = static_cast<Complex<OutputFloat>*>(r0);
        outputDotArg.input2 = static_cast<Complex<OutputFloat>*>(r_new);
        outputDotArg.resArr = static_cast<Complex<OutputFloat>*>(rho_new);
        interior_operator_.output_dotc(outputDotArg);

        // beta_temp = alpha * rho_new / omega / rho_i
        // step 1 : beta = alpha * <r0, r_new> = alpha * rho_new
        output_elementwise_mul_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
        output_elementwise_mul_arg.x       = static_cast<Complex<OutputFloat>*>(alpha_array);
        output_elementwise_mul_arg.y       = static_cast<Complex<OutputFloat>*>(rho_new);
        interior_operator_.output_elementwise_mul(output_elementwise_mul_arg);

        // step 2 : beta = beta / omega = alpha * rho_new / omega
        output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
        output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(beta_array);
        output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(omega_array);
        interior_operator_.output_elementwise_div(output_elementwise_div_arg);

        // third step : beta = beta / rho_i
        output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
        output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(beta_array);
        output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(rho_j);
        interior_operator_.output_elementwise_div(output_elementwise_div_arg);

        // p_new = r_new + beta * (pj - omega * Ap)
        // first step: p_new = pj - omega * A pj
        output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(p_new);
        output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(pj);
        output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(omega_array);
        output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(vj);
        interior_operator_.output_xsay(output_xsay_arg); // s = pj - omega * Ap

        // second step: p_new = r_new + beta * p_new
        output_xpay_arg.res = static_cast<Complex<OutputFloat>*>(p_new);
        output_xpay_arg.x   = static_cast<Complex<OutputFloat>*>(r_new);
        output_xpay_arg.a   = static_cast<Complex<OutputFloat>*>(beta_array);
        output_xpay_arg.y   = static_cast<Complex<OutputFloat>*>(p_new);
        output_xpay_arg.stream = stream1;
        interior_operator_.output_xpay(output_xpay_arg); // p_new = r_new + beta * p_new

        std::swap(rj, r_new);  // rj = r_new
        std::swap(pj, p_new);  // pj = p_new
    }

    return currentIteration_ < maxIteration_ && isConverged(norm_r_array, norm_b_array, maxPrec_);
}
template <
    QcuPrecision OutputPrecision,
    QcuPrecision IteratePrecision
>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve_odd_policy2() {
  std::cout << "POLICY2 BICGStab: Combined Residual" << std::endl;
  OutputFloat norm_r = OutputFloat(1.0);
  OutputFloat norm_b = OutputFloat(1.0);

  // diff_array = [r1, r2, r3, ...] / [b1, b2, b3, ...]

  const int mInput = param_.mInput;
  const int vol = param_.lattDesc->lattice_volume();
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
  void*       rj = outputBuffer_[10]; // 初始时 r0 == rj   因为要求<r0, rj> 不为0
  void*       pj = outputBuffer_[11]; // pj = r0，从此outputBuffer[1]不变，不能再使用

  void* vj           = outputBuffer_[1];  // 每轮要留住 outputBuffer_[1]作为 vj = Ap
  void* t            = outputBuffer_[2];  // 每轮要留住 outputBuffer_[2]作为 t = As
  void* sj           = outputBuffer_[3];
  void* r_new        = outputBuffer_[4];
  void* reduceBuffer = outputBuffer_[5];  // use this tempBuffer to reduce
  void* x_new        = outputBuffer_[6];
  void* p_new        = outputBuffer_[7];
  // void* x            = outputBuffer_[8];
  void* x_o          = static_cast<Complex<OutputFloat>*>(result_x_output_prec_) + vol / 2 * complex_vec_len;
  void* temp_buffer  = outputBuffer_[9];

  void* kappa_square_array = output_scala_array_[5];
  void* one_array          = output_scala_array_[1];
  void* rho_j              = output_scala_array_[2]; // 要用到最后 rho_i = <r0, ri>
  void* r0_dot_vj          = output_scala_array_[3]; // r0_dot_vj = <r0, vj> = <r0, A pj>
  void* rho_new            = output_scala_array_[4]; // rho_new = <r0, r_{j+1}>
  void* t_dot_sj           = output_scala_array_[8]; // t_dot_sj = <As, sj>
  void* t_dot_t            = output_scala_array_[6]; // t_dot_t = <As, As>
  void* r_new_norm         = output_scala_array_[7];
  using OutputNormArgument = typename InteriorOperator::template ComplexNorm<OutputFloat, OutputFloat>
                                                      ::template ComplexNormArgument;
  OutputNormArgument output_norm_arg {
    vol * single_complex_vec_len / 2 * mInput,              // single_vec_len;
    1,                                                      // stride;
    static_cast<OutputFloat*>(temp_buffer),                 // tmpBuffer
    static_cast<Complex<OutputFloat>*>(b),                  // input
    static_cast<OutputFloat*>(output_new_b_even_norm),      // resArr
    stream1,
    cublasHandle_
  };

  interior_operator_.output_norm(output_norm_arg);
  // 计算norm，一次性保存到host端
  CHECK_CUDA(cudaMemcpyAsync(&norm_b, output_new_b_even_norm, sizeof(OutputFloat), cudaMemcpyDeviceToHost, stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream1));

  // R = b - A * x = b - Dslash * x, x可以初始化为0
  std::shared_ptr<DslashParam> dslashParam = std::make_shared<DslashParam>(
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
  );

  using Output_xsayArgument = typename InteriorOperator::template Complex_xsay<OutputFloat>
                                                       ::template Complex_xsayArgument;

  Output_xsayArgument output_xsay_arg { nullptr, nullptr, nullptr, nullptr,
      vol * single_complex_vec_len / 2 * mInput,      // int single_vec_len,
      1,                                              // int inc_idx,
      stream1
  };

  // InnerProduct Param
  using OutputDotcArgument = typename InteriorOperator::template ComplexDotc<OutputFloat, OutputFloat>
                                                      ::template DotcArgument;
  OutputDotcArgument outputDotArg {
    vol * single_complex_vec_len / 2 * mInput, // single_vec_len;
    1,                                // stride;
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
  OutputElementwiseDivArgument output_elementwise_div_arg { nullptr, nullptr, nullptr, 1 /* vec_len */ };

  // ElementwiseMul Param
  using OutputElementwiseMulArgument = typename InteriorOperator::template ElementwiseMul<Complex<OutputFloat>>
                                                                ::template ElementwiseMulArgument;
  OutputElementwiseMulArgument output_elementwise_mul_arg { nullptr, nullptr, nullptr, 1 /* vec_len */ };

  using OutputAxpbypczArgument = typename InteriorOperator::template Complex_axpbypcz<OutputFloat>
                                                          ::template Complex_axpbypczArgument;
  OutputAxpbypczArgument output_axpbypcz_arg {nullptr, nullptr, nullptr,nullptr, nullptr, nullptr, nullptr,
    vol * single_complex_vec_len / 2 * mInput,  /* single_vec_len */
    1,                                          /* inc_idx */
    stream1                                     /* stream*/
  };

  using Output_xpayArgument = typename InteriorOperator::template Complex_xpay<OutputFloat>
                                                        ::template Complex_xpayArgument;
  Output_xpayArgument output_xpay_arg {
      nullptr, nullptr, nullptr, nullptr,
      vol * single_complex_vec_len / 2 * mInput, /*single_vec_len*/
      1                            /*inc_idx*/,
      stream1 };

  // prelogue
  // x = 0
  CHECK_CUDA(cudaMemsetAsync(x_new, vol / 2 * complex_vec_len, 0, stream1));
  // rj = pj = r0 = b - Ax = b
  CHECK_CUDA(cudaMemcpyAsync(rj, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
  CHECK_CUDA(cudaMemcpyAsync(r0, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
  CHECK_CUDA(cudaMemcpyAsync(pj, b, sizeof(Complex<OutputFloat>) * complex_vec_len * vol / 2, cudaMemcpyDeviceToDevice, stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream1));

  // begin iteration
  // 开始迭代，达到最大迭代次数不收敛则返回false
  for (currentIteration_ = 0; currentIteration_ < maxIteration_; ++ currentIteration_) {
    // rho_j = <r0, r_j>
    outputDotArg.input1    = static_cast<Complex<OutputFloat>*>(r0);
    outputDotArg.input2    = static_cast<Complex<OutputFloat>*>(rj);
    outputDotArg.resArr    = static_cast<Complex<OutputFloat>*>(rho_j);
    outputDotArg.tmpBuffer = static_cast<Complex<OutputFloat>*>(reduceBuffer);
    interior_operator_.output_dotc(outputDotArg);

    // vj = Ap = Ap_{j} = Doe Deo * p_{j} ----> outputBuffer_[1];
    fused_x_sub_Doe_Deo_x<OutputFloat>(vj, pj, temp_buffer, kappa_square_array, dslash_operator_, dslashParam);
    cudaStreamSynchronize(dslashParam->stream1);
    cudaStreamSynchronize(dslashParam->stream2);

    // r0_dot_vj = <r0, vj> = <r0, Ap_j>
    // , norm <r0, Ap>
    outputDotArg.input2    = static_cast<Complex<OutputFloat>*>(vj);
    outputDotArg.resArr    = static_cast<Complex<OutputFloat>*>(r0_dot_vj);
    interior_operator_.output_dotc(outputDotArg);

    // alpha = <r0, r_i> / <r0, A pj> = rho_i / r0_dot_vj
    output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(alpha_array);
    output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(rho_j);
    output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(r0_dot_vj);
    interior_operator_.output_elementwise_div(output_elementwise_div_arg);
    CHECK_CUDA(cudaStreamSynchronize(output_elementwise_div_arg.stream));

    // sj = rj - alpha_{j} * vj = rj - alpha_{j} * A pj
    output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(sj);
    output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(rj);
    output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(alpha_array);
    output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(vj);
    interior_operator_.output_xsay(output_xsay_arg);

    // t = A sj = Doe Deo * sj
    fused_x_sub_Doe_Deo_x<OutputFloat>(t, sj, temp_buffer, kappa_square_array, dslash_operator_, dslashParam);

    // omega = <As, s> / <As, As> = t_dot_sj / t_dot_t
    // step1:  t_dot_sj = <As, s> = <t, sj>
    outputDotArg.input1 = static_cast<Complex<OutputFloat>*>(t);
    outputDotArg.input2 = static_cast<Complex<OutputFloat>*>(sj);
    outputDotArg.resArr = static_cast<Complex<OutputFloat>*>(t_dot_sj);
    interior_operator_.output_dotc(outputDotArg);
    // step2: <As, As>     -----> output_scala_array_[3]
    outputDotArg.input2  = static_cast<Complex<OutputFloat>*>(t);
    outputDotArg.resArr  = static_cast<Complex<OutputFloat>*>(t_dot_t);
    interior_operator_.output_dotc(outputDotArg);
    // step3: omega = <As, s> / <As, As>
    output_elementwise_div_arg.res = static_cast<Complex<OutputFloat>*>(omega_array);
    output_elementwise_div_arg.x   = static_cast<Complex<OutputFloat>*>(t_dot_sj);
    output_elementwise_div_arg.y   = static_cast<Complex<OutputFloat>*>(t_dot_t);
    interior_operator_.output_elementwise_div(output_elementwise_div_arg);

    // x_new = x + alpha * pj + omega * sj
    output_axpbypcz_arg.res = static_cast<Complex<OutputFloat>*>(x_new);
    output_axpbypcz_arg.a   = static_cast<Complex<OutputFloat>*>(one_array);
    output_axpbypcz_arg.x   = static_cast<Complex<OutputFloat>*>(x_new);
    output_axpbypcz_arg.b   = static_cast<Complex<OutputFloat>*>(alpha_array);
    output_axpbypcz_arg.y   = static_cast<Complex<OutputFloat>*>(pj);
    output_axpbypcz_arg.c   = static_cast<Complex<OutputFloat>*>(omega_array);
    output_axpbypcz_arg.z   = static_cast<Complex<OutputFloat>*>(sj);
    output_axpbypcz_arg.stream = stream1;
    interior_operator_.output_axpbypcz(output_axpbypcz_arg); // x_new = x + alpha * pj + omega * sj

    // r_new = s - omega * As = s - omega t
    output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(r_new);
    output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(sj);
    output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(omega_array);
    output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(t);
    interior_operator_.output_xsay(output_xsay_arg);

    // if converge ?
    { // converge return true
      // calculate norm of r_new and store them
      // void* r_new_norm = output_scala_array_[3];
      output_norm_arg.input  = static_cast<Complex<OutputFloat>*>(r_new);
      output_norm_arg.resArr = static_cast<OutputFloat*>(r_new_norm);
      interior_operator_.output_norm(output_norm_arg); // 计算norm，一次性保存到host端
      CHECK_CUDA(cudaMemcpyAsync(&norm_r, r_new_norm, sizeof(OutputFloat), cudaMemcpyDeviceToHost, stream1));
      CHECK_CUDA(cudaStreamSynchronize(stream1));
// #ifdef DEBUG
//       std::printf("DEBUG, currentIteration = %d\n", currentIteration_);
// #endif
      if (bool is_converged = isConverged_policy2<OutputFloat>(norm_r, norm_b, maxPrec_ /*/ std::sqrt(OutputFloat(mInput))*/)) {
        CHECK_CUDA(cudaMemcpyAsync(x_o, x_new, sizeof(OutputFloat) * vol / 2 * complex_vec_len * 2,
                              cudaMemcpyDeviceToDevice, stream1)); // res_x = x_new = x_{j + 1}
        CHECK_CUDA(cudaStreamSynchronize(stream1));
        return true;
      }
    }
    // beta =  (alpha / omega)(<r0, r_new> / <r0, rj>) = (alpha / omega) (rho_new / rho_i)
    // we now have <r, r0> in rho_i
    // now calculate <r0, r_new> and store it in rho_new
    outputDotArg.input1 = static_cast<Complex<OutputFloat>*>(r0);
    outputDotArg.input2 = static_cast<Complex<OutputFloat>*>(r_new);
    outputDotArg.resArr = static_cast<Complex<OutputFloat>*>(rho_new);
    interior_operator_.output_dotc(outputDotArg);

    // beta_temp = alpha * rho_new / omega / rho_i
    // step 1 : beta = alpha * <r0, r_new> = alpha * rho_new
    output_elementwise_mul_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_mul_arg.x       = static_cast<Complex<OutputFloat>*>(alpha_array);
    output_elementwise_mul_arg.y       = static_cast<Complex<OutputFloat>*>(rho_new);
    interior_operator_.output_elementwise_mul(output_elementwise_mul_arg);

    // step 2 : beta = beta / omega = alpha * rho_new / omega
    output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(omega_array);
    interior_operator_.output_elementwise_div(output_elementwise_div_arg);

    // third step : beta = beta / rho_i
    output_elementwise_div_arg.res     = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.x       = static_cast<Complex<OutputFloat>*>(beta_array);
    output_elementwise_div_arg.y       = static_cast<Complex<OutputFloat>*>(rho_j);
    interior_operator_.output_elementwise_div(output_elementwise_div_arg);

    // p_new = r_new + beta * (pj - omega * Ap)
    // first step: p_new = pj - omega * A pj
    output_xsay_arg.res = static_cast<Complex<OutputFloat>*>(p_new);
    output_xsay_arg.x   = static_cast<Complex<OutputFloat>*>(pj);
    output_xsay_arg.a   = static_cast<Complex<OutputFloat>*>(omega_array);
    output_xsay_arg.y   = static_cast<Complex<OutputFloat>*>(vj);
    interior_operator_.output_xsay(output_xsay_arg); // s = pj - omega * Ap

    // second step: p_new = r_new + beta * p_new
    output_xpay_arg.res = static_cast<Complex<OutputFloat>*>(p_new);
    output_xpay_arg.x   = static_cast<Complex<OutputFloat>*>(r_new);
    output_xpay_arg.a   = static_cast<Complex<OutputFloat>*>(beta_array);
    output_xpay_arg.y   = static_cast<Complex<OutputFloat>*>(p_new);
    output_xpay_arg.stream = stream1;
    interior_operator_.output_xpay(output_xpay_arg); // p_new = r_new + beta * p_new

    std::swap(rj, r_new);  // rj = r_new
    std::swap(pj, p_new);  // pj = p_new
  }

  return currentIteration_ < maxIteration_ && isConverged_policy2(norm_r, norm_b, maxPrec_);
}

template <
    QcuPrecision OutputPrecision,
    QcuPrecision IteratePrecision
>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve_odd() {
    return solve_odd_policy1();
}
template <
    QcuPrecision OutputPrecision,
    QcuPrecision IteratePrecision
>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve_even() {

    const int mInput = param_.mInput;
    const int vol = param_.lattDesc->lattice_volume();
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
    std::shared_ptr<DslashParam> dslashParam = std::make_shared<DslashParam>(
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
    );

    dslash_operator_->apply(dslashParam);  // x_e = D_{eo} x_{o}

    CHECK_CUDA(cudaDeviceSynchronize());

    // x_e = b_e + kappa D_{eo} x_{o}
    //     = b_e + kappa x_e
    // using Output_xpayArgument = typename InteriorOperator::Output_xpayAruArgument;
    using Output_xpayArgument =
        typename InteriorOperator::template Complex_xpay<OutputFloat>::template Complex_xpayArgument;
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

template <
    QcuPrecision OutputPrecision,
    QcuPrecision IteratePrecision
>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve() {
    if (!bufferAllocated_) {
        if (!tempBufferAllocate()) {
            return false;
        }
    }

    if (!solve_odd()) {
        printf("QCU BICGStab solve odd failed, %d iterations\n", currentIteration_);
        return false;
    }
    if (!solve_even()) {
        printf("QCU BICGStab solve even failed, %d iterations\n", currentIteration_);
        return false;
    }

    printf("QCU BICGStab solve success, %d iterations\n", currentIteration_);
    const int vol = param_.lattDesc->lattice_volume();
    const int mrhs_vec_len = param_.mInput * param_.nColor * Ns; // on single point
    const cudaStream_t cuda_stream = param_.stream1;
    // copy x to outputBuffer
    copyComplexVector_interface(
        param_.output_x_mrhs, OutputPrecision,
        result_x_output_prec_, OutputPrecision,
        vol * mrhs_vec_len, cuda_stream);

    CHECK_CUDA(cudaDeviceSynchronize());
    return true;
}

// donnot use HALF to be the output precision
template class BiCGStabImpl<QcuPrecision::kPrecisionDouble, QcuPrecision::kPrecisionDouble>;
template class BiCGStabImpl<QcuPrecision::kPrecisionDouble, QcuPrecision::kPrecisionSingle>;
template class BiCGStabImpl<QcuPrecision::kPrecisionDouble, QcuPrecision::kPrecisionHalf>;
template class BiCGStabImpl<QcuPrecision::kPrecisionSingle, QcuPrecision::kPrecisionDouble>;
template class BiCGStabImpl<QcuPrecision::kPrecisionSingle, QcuPrecision::kPrecisionSingle>;
template class BiCGStabImpl<QcuPrecision::kPrecisionSingle, QcuPrecision::kPrecisionHalf>;
}