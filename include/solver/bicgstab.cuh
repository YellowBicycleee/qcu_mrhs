#pragma once

#include <cublas_v2.h>
#include "base/datatype/qcu_float2.cuh"
#include "desc/qcu_desc.h"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_blas/qcu_blas.h"
#include "qcu_public.h"
#include "qcu_config/qcu_config.h"
namespace qcu::solver {

struct BiCGStabParam {
    int nColor;
    int mInput;
    int Nspin;
    int t_boudary;   // add attribute
    QcuStaggeredPhase staggered_phase = QcuStaggeredPhase::kQcuStaggeredPhaseCps;  // add attribute
    double kappa;
    void* output_x_mrhs;
    void* input_b_mrhs;
    void* gauge;
    const QcuLattDesc* lattDesc;
    const QcuProcDesc* procDesc;
    std::vector<cudaStream_t>& streams;
    std::shared_ptr<qcu::FermionGhost<Nd>> fermion_ghost_;
    bool use_combined_residual = false;
    bool use_tensor_core = false;
};

// OutputPrecision     既表示输入又表示输出精度，
// IteratePrecision    表示迭代精度
template <
    QcuPrecision ReducePrecision,
    QcuPrecision ComputePrecision
>
class BiCGStabImpl {
public:
    BiCGStabImpl () = delete;
    BiCGStabImpl (BiCGStabParam& param, int max_iteration = 1000, double max_precision = 1e-6)
        : param_(param)
        , maxIteration_(max_iteration)
        , maxPrec_(ReduceFloat(max_precision))
    {
        tempBufferAllocate();
    }
    ~BiCGStabImpl() noexcept {
        tempBufferFree();
    }
    bool solve();  // return true if converged
    // out = in - a DoeDeo in
    template <typename _Float>
    inline void fused_x_sub_Doe_Deo_x (
        void* output, void* input, void* temp, void* a,
        std::shared_ptr<qcu::Dslash> dslash,
        std::shared_ptr<qcu::DslashParam> param)
    {
        const int vol = qcu::config::lattice_volume_local();
        const int m_input = param->m_input;
        const int n_color = param->n_color;
        const int single_vec_len = Nd * n_color;

        cudaStream_t stream1 = param->streams[8];
        cudaStream_t stream2 = param->streams[7];
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
private:
    bool solve_odd();
    bool solve_odd_separated_residual(); // 单独计算norm和内积
    bool solve_odd_combined_residual(); // 所有残差按一个计算
    bool solve_even();
    using ReduceFloat  = typename qcu::Float2WrapperFromPrecision<ReducePrecision>::Float;
    using ReduceFloat2 = typename qcu::Float2_t<ReduceFloat>;
    using ComputeFloat = typename qcu::Float2WrapperFromPrecision<ComputePrecision>::Float;
    using ComputeFloat2= typename qcu::Float2_t<ComputeFloat>;

    struct InteriorOperator {
        // operator
        // 复数内积运算符
        template <typename ReduceFloat_, typename ComputeFloat_>
        using ComplexDotc         = typename qcu::qcu_blas::ComplexDotc<ReduceFloat_, ComputeFloat_>;
        ComplexDotc<ReduceFloat, ComputeFloat> output_dotc;

        // norm2 运算符
        template <typename ReduceFloat_, typename ComputeFloat_>
        using ComplexNorm        = typename qcu::qcu_blas::ComplexNorm<ReduceFloat_, ComputeFloat_>;
        ComplexNorm<ReduceFloat, ComputeFloat>  output_norm;

        // xpay 运算符
        template<typename ComputeFloat_>
        using Complex_xpay        = typename qcu::qcu_blas::Complex_xpay<ComputeFloat_>;
        Complex_xpay<ComputeFloat>  output_xpay;

        // xsay 运算符
        template <typename ComputeFloat_>
        using Complex_xsay = typename qcu::qcu_blas::Complex_xsay<ComputeFloat_>;
        Complex_xsay<ComputeFloat>  output_xsay;

        // axpby运算符
        template <typename ComputeFloat_>
        using Complex_axpby = typename qcu::qcu_blas::Complex_axpby<ComputeFloat_>;
        Complex_axpby<ComputeFloat>  output_axpby;

        // axpbypcz运算符
        template <typename ComputeFloat_>
        using Complex_axpbypcz       = typename qcu::qcu_blas::Complex_axpbypcz<ComputeFloat_>;
        Complex_axpbypcz<ComputeFloat>  output_axpbypcz;

        // elementwise_div 运算符
        template <typename Tp_>
        using ElementwiseDiv                   = typename qcu::qcu_blas::ElementwiseDiv<Tp_>;
        ElementwiseDiv<Complex<ReduceFloat>>   output_elementwise_div;

        // elementwise_mul 运算符
        template <typename Tp_>
        using ElementwiseMul                   = typename qcu::qcu_blas::ElementwiseMul<Tp_>;
        ElementwiseMul<Complex<ReduceFloat>>   output_elementwise_mul;   // 迭代 elementwise_div 运算符

        // elementwise_init 运算符
        template <typename Tp_> using ElementwiseInit = typename qcu::qcu_blas::ElementwiseInit<Tp_>;
        ElementwiseInit<Complex<ReduceFloat>>  output_elementwise_init; // 高精度 elementwise_init 运算符

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
    ReduceFloat  maxPrec_          = 1e-6;

    std::shared_ptr<Dslash>  dslash_operator_      = nullptr;

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