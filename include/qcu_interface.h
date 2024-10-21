#pragma once

#include <vector>
#include <cstdint>

#include "desc/qcu_desc.h"
#include "qcd/qcu_dslash.h"
#include "qcu_public.h"


namespace qcu {
class Qcu {
public:
    struct Argument {
        int32_t n_color;    // number of colors
        int32_t m_rhs;      // number of right hand side
        double mass; // kappa = 1 / (2 * (4 + mass))
        QCU_PRECISION out_float_precision;
        QCU_PRECISION compute_float_precision;
    };
private:
    int32_t n_colors_;
    int32_t m_input_;
    double mass_;
    double kappa_;
    QCU_PRECISION out_float_precision_; // use it as input and output precision
    QCU_PRECISION compute_floatprecision_;// use it as calculation precision such as dslash and solver

    QcuLattDesc lattice_desc_;
    QcuProcDesc process_desc_;
    DslashParam *dslash_param_;
    Dslash *dslash_;

    std::vector<void *> fermion_in_queue_;
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
        QCU_PRECISION outputFloatPrecision,
        QCU_PRECISION iterateFloatPrecision = QCU_DOUBLE_PRECISION,
        int nColors = 3, int mInputs = 1, double mass = 0.0,
        bool inverterEnabled = false)
        : n_colors_(nColors),
          m_input_(mInputs),
          mass_(mass),
          kappa_(1.0 / (2.0 * (4.0 + mass))),
          lattice_desc_(Lx, Ly, Lz, Lt),
          process_desc_(Gx, Gy, Gz, Gt),
          out_float_precision_(outputFloatPrecision),
          dslash_param_(nullptr),
          dslash_(nullptr),
          gauge_external_(nullptr),
          fp64_gauge_(nullptr),
          fp32_gauge_(nullptr),
          fp16_gauge_(nullptr),
          fermion_in_mrhs_(nullptr),
          fermion_out_mrhs_(nullptr),
          compute_floatprecision_(iterateFloatPrecision)
    {
        allocateMemory();
    }

    ~Qcu() { freeMemory(); }

    QcuLattDesc lattice_desc() const { return lattice_desc_; }
    QcuProcDesc process_desc() const { return process_desc_; }
    
    int32_t color() const { return n_colors_; }
    int32_t rhs_num () const { return m_input_; }
    int32_t n_spin () const { return Ns; }

    void get_dslash (DSLASH_TYPE dslashType, double mass);
    void start_dslash (int parity, bool daggerFlag = false);
    void mat_qcu (bool daggerFlag = false);
    void load_gauge (void *gauge, QCU_PRECISION floatPrecision);

    void push_back_fermion (void *fermionOut, void *fermionIn);
    // solve Ax = b
    void solve_fermions (int max_iteration, double p_max_prec);
    // IO
    void read_gauge_from_file (const char* file_path, void* data_ptr);
};

}  // namespace qcu