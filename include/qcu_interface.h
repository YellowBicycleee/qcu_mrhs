#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "desc/qcu_desc.h"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_helper_macro.h"
#include "qcu_public.h"
#include "qcd/qcu_dslash_staggered.h"
#include <cuda_fp16.h>
namespace qcu {
class Qcu {
public:
    struct Argument {
        int32_t n_color;    // number of colors
        int32_t m_rhs;      // number of right hand side
        double mass;        // kappa = 1 / (2 * (4 + mass))
        double kappa;
        QcuPrecision out_float_precision;
        QcuPrecision compute_float_precision;
        qcu::QcuLattDesc lattice_desc_ptr;
        qcu::QcuProcDesc process_desc_ptr;

        Argument (int32_t n_color_, int m_rhs_, double mass_, double kappa_,
            QcuPrecision out_float_precision_, QcuPrecision compute_float_precision_,
            qcu::QcuLattDesc lattice_desc_ptr_, qcu::QcuProcDesc process_desc_ptr_)
        : n_color(n_color_), m_rhs(m_rhs_), mass(mass_), kappa(kappa_),
          out_float_precision(out_float_precision_), compute_float_precision(compute_float_precision_),
          lattice_desc_ptr(lattice_desc_ptr_), process_desc_ptr(process_desc_ptr_)
        {}
    };
private:
    Argument underlying_args_;

    int32_t n_colors_;
    int32_t m_input_;
    int32_t n_spin_ = -1;
    int32_t t_boundary_ = 1;
    bool anti_periodic_t_ = false;
    double mass_;
    double kappa_;
    bool tensor_core_flag_ = false;
    bool residual_combine_flag_ = false;
    QcuStaggeredPhase staggered_phase_ = QcuStaggeredPhase::kQcuStaggeredPhaseNo;

    std::shared_ptr<DslashParam> dslash_param_ = nullptr;
    std::shared_ptr<Dslash> dslash_ = nullptr;

    std::vector<void *> fermion_in_vec_;
    std::vector<void *> fermion_out_vec_;

    void *gauge_external_ = nullptr;      // gauge field, donnot allocate memory, external pointer
    void *fp64_gauge_ = nullptr;          // double gauge field
    void *fp32_gauge_ = nullptr;          // single gauge field
    void *fp16_gauge_ = nullptr;          // half gauge field

    // mrhs fermion field, gathered into my preferred shape
    void *fermion_in_mrhs_ = nullptr;
    void *fermion_out_mrhs_ = nullptr;

    // lookup table
    void* d_lookup_table_in_ = nullptr;
    void* d_lookup_table_out_ = nullptr;

    void* device_kappa_ = nullptr;

    void* cpu_allocator_ = nullptr; // TODO: add allocator, reserved for future use
    void* gpu_allocator_ = nullptr; // TODO: add allocator, reserved for future use

    void allocateMemory();

    void freeMemory();

    // template<typename OutputFloat>
    // void solve_fermions_template_function (int max_iteration, double p_max_prec);

    template <typename ComputeFloat_, typename ReduceFloat_>
    void mat_qcu_template_function (bool dagger_flag);
public:
    Qcu(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt,
        QcuPrecision outputFloatPrecision,
        QcuPrecision computeFloatPrecision = QcuPrecision::kPrecisionDouble,
        int nColors = 3, int mInputs = 1, double mass = 0.0,
        bool inverterEnabled = false)
        : n_colors_(nColors)
        , m_input_(mInputs)
        , mass_(mass)
        , kappa_(1.0 / (2.0 * (4.0 + mass)))
        , underlying_args_(nColors, mInputs, mass,
                                1.0 / (2.0 * (4.0 + mass)),
                                outputFloatPrecision,
                                computeFloatPrecision,
                                qcu::QcuLattDesc{Lx, Ly, Lz, Lt},
                                qcu::QcuProcDesc{Gx, Gy, Gz, Gt}
            )
        , dslash_param_(nullptr)
        , dslash_(nullptr)
        , gauge_external_(nullptr)
        , fp64_gauge_(nullptr)
        , fp32_gauge_(nullptr)
        , fp16_gauge_(nullptr)
        , fermion_in_mrhs_(nullptr)
        , fermion_out_mrhs_(nullptr)
    {}

    ~Qcu() { freeMemory(); /* getDslash申请内存 */ }

    int32_t color() const { return n_colors_; }
    int32_t rhs_num () const { return m_input_; }

    void get_dslash (DslashType dslashType, double mass, bool anti_periodic_t);
    void start_dslash (int parity, bool daggerFlag = false);
    void mat_qcu (bool daggerFlag = false);
    void load_gauge (void *gauge, QcuPrecision floatPrecision);

    void push_back_fermion (void *fermionOut, void *fermionIn);
    // solve Ax = b

    void solve_fermions (int max_iteration, double p_max_prec);
    // IO
    // template <typename Float_>
    template <typename Float_>
    void read_gauge_from_file (const char* file_path, void* data_ptr);

    void set_staggered_phase (QcuStaggeredPhase staggered_phase);
    QcuPrecision io_precision() const { return underlying_args_.out_float_precision; }
    QcuPrecision compute_precision() const { return underlying_args_.compute_float_precision; }

    // 启动scatter
    void begin_scatter();
    // 启动gather
    void begin_gather();
    void set_tensor_core_flag(int tensor_core_flag) {
        if (tensor_core_flag) {
            tensor_core_flag_ = true;
        } else {
            tensor_core_flag_ = false;
        }
    }
    void set_residual_combine_flag(int residual_combine_flag) {
        if (residual_combine_flag) {
            residual_combine_flag_ = true;
        } else {
            residual_combine_flag_ = false;
        }
    }
};




}  // namespace qcu