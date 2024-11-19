#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "desc/qcu_desc.h"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_helper_macro.h"
#include "qcu_public.h"
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
    double mass_;
    double kappa_;

    std::shared_ptr<DslashParam> dslash_param_ = nullptr;
    std::shared_ptr<Dslash> dslash_ = nullptr;

    std::vector<void *> fermion_in_vec_;
    std::vector<void *> fermion_out_vec_;

    void *gauge_external_;      // gauge field, donnot allocate memory, external pointer
    void *fp64_gauge_;          // double gauge field
    void *fp32_gauge_;          // single gauge field
    void *fp16_gauge_;          // half gauge field

    // mrhs fermion field, gathered into my preferred shape
    void *fermion_in_mrhs_;
    void *fermion_out_mrhs_;

    // lookup table
    void* d_lookup_table_in_;
    void* d_lookup_table_out_;

    void* device_kappa_ = nullptr;

    void* cpu_allocator_ = nullptr; // TODO: add allocator, reserved for future use
    void* gpu_allocator_ = nullptr; // TODO: add allocator, reserved for future use

    void allocateMemory();
    void freeMemory();

public:
    Qcu(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt,
        QcuPrecision outputFloatPrecision,
        QcuPrecision iterateFloatPrecision = QcuPrecision::kPrecisionDouble,
        int nColors = 3, int mInputs = 1, double mass = 0.0,
        bool inverterEnabled = false)
        : n_colors_(nColors)
        , m_input_(mInputs)
        , mass_(mass)
        , kappa_(1.0 / (2.0 * (4.0 + mass)))
        , underlying_args_(nColors, mInputs, mass,
                                1.0 / (2.0 * (4.0 + mass)),
                                outputFloatPrecision,
                                iterateFloatPrecision,
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
    {
        allocateMemory();
    }

    ~Qcu() { freeMemory(); }

    int32_t color() const { return n_colors_; }
    int32_t rhs_num () const { return m_input_; }
    int32_t n_spin () const { return Ns; }

    void get_dslash (DslashType dslashType, double mass);
    void start_dslash (int parity, bool daggerFlag = false);
    void mat_qcu (bool daggerFlag = false);
    void load_gauge (void *gauge, QcuPrecision floatPrecision);

    void push_back_fermion (void *fermionOut, void *fermionIn);
    // solve Ax = b
    void solve_fermions (int max_iteration, double p_max_prec);
    // IO
    void read_gauge_from_file (const char* file_path, void* data_ptr);
};

}  // namespace qcu