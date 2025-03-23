#include <qcu_config/qcu_config.h>
#include <iostream>
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



namespace qcu::solver {

template <
    QcuPrecision OutputPrecision,
    QcuPrecision IteratePrecision
>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve_odd() {
    if (param_.use_combined_residual) {
        return solve_odd_combined_residual();
    }
    else {
        return solve_odd_separated_residual();
    }
}
template <
    QcuPrecision OutputPrecision,
    QcuPrecision IteratePrecision
>
bool BiCGStabImpl<OutputPrecision, IteratePrecision>::solve_even() {

    const int mInput = param_.mInput;
    const int vol = param_.lattDesc->lattice_volume();
    const int single_complex_vec_len = param_.nColor * param_.Nspin;
    const int mrhs_complex_vec_len = param_.mInput * single_complex_vec_len;
    // solve x_e
    // x_e = b_e + kappa D_{eo} x_{o}
    // now we get x_{o} and b_{e}
    void* x_e = result_x_output_prec_;
    void* x_o = static_cast<Complex<ComputeFloat>*>(x_e) + vol / 2 * mrhs_complex_vec_len;
    void* b_e_origin = param_.input_b_mrhs;;
    void* kappa_array = output_scala_array_[0];

    // D_{eo} x_{o} ----> x_e
    std::shared_ptr<DslashParam> dslashParam = std::make_shared<DslashParam>(
        false,                      // bool p_daggerFlag,
        OutputPrecision,            // QCU_PRECISION p_precision,
        param_.staggered_phase,     // int p_staggered_phase,
        param_.t_boudary,           // int p_t_boundary,
        param_.nColor,              // int p_nColor,
        param_.mInput,              // int p_mInput,
        EVEN_PARITY,                // int p_parity,
        ReduceFloat(param_.kappa),  // double p_kappa,
        x_o,                        // void* p_fermionIn_MRHS
        x_e,                        // void* p_fermionOut_MRHS,
        param_.gauge,               // void* p_gauge,
        param_.lattDesc,            // const QcuLattDesc* p_lattDesc,
        param_.procDesc,            // const QcuProcDesc* p_procDesc,
        param_.streams,
        param_.fermion_ghost_
    );

    dslash_operator_->apply(dslashParam);  // x_e = D_{eo} x_{o}

    CHECK_CUDA(cudaDeviceSynchronize());

    // x_e = b_e + kappa D_{eo} x_{o}
    //     = b_e + kappa x_e
    // using Output_xpayArgument = typename InteriorOperator::Output_xpayAruArgument;
    using Output_xpayArgument =
        typename InteriorOperator::template Complex_xpay<ComputeFloat>::template Complex_xpayArgument;
    Output_xpayArgument output_xpay_arg {
        static_cast<Complex<ComputeFloat>*>(x_e),          // Complex<_Float>* res,
        static_cast<Complex<ComputeFloat>*>(b_e_origin),          // Complex<_Float>* x,
        static_cast<Complex<ComputeFloat>*>(kappa_array),  // Complex<_Float>* a,
        static_cast<Complex<ComputeFloat>*>(x_e),          // Complex<_Float>* y,
        vol / 2 * single_complex_vec_len,                 // int single_vec_len,
        mInput,                                           // int inc_idx,
        param_.streams[8]
    };
    interior_operator_.output_xpay(output_xpay_arg); // x_e = b_e + kappa x_e
    CHECK_CUDA(cudaStreamSynchronize(param_.streams[8]));
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
    const int mrhs_vec_len = param_.mInput * param_.nColor * param_.Nspin; // on single point
    const cudaStream_t cuda_stream = param_.streams[8];
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
template class BiCGStabImpl<QcuPrecision::kPrecisionSingle, QcuPrecision::kPrecisionSingle>;
template class BiCGStabImpl<QcuPrecision::kPrecisionSingle, QcuPrecision::kPrecisionHalf>;
}