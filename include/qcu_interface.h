#pragma once
// #include <cuda_fp16.h>

#include <vector>

#include "desc/qcu_desc.h"
#include "qcd/qcu_dslash.h"
#include "qcu_public.h"
#include <cstdint>

namespace qcu {
class Qcu {
    bool inverterEnabled_;
    int32_t nColors_;
    int32_t mInput_;
    double mass_;
    double kappa_;
    QCU_PRECISION outputFloatPrecision_; // use it as input and output precision
    QCU_PRECISION iterateFloatPrecision_;// use it as calculation precision such as dslash and solver

    QcuLattDesc lattDesc_;
    QcuProcDesc procDesc_;
    DslashParam *dslashParam_;
    Dslash *dslash_;

    std::vector<void *> fermionIn_queue_;
    std::vector<void *> fermionOut_queue_;

    void *gauge_;      // gauge field, donnot allocate memory, external pointer
    void *fp64Gauge_;  // double gauge field
    void *fp32Gauge_;  // single gauge field
    void *fp16Gauge_;  // half gauge field

    // mrhs fermion field
    void *fermionIn_MRHS_;  // also used as b in Ax=b, to solve x
    void *fermionOut_MRHS_; // also used as x in Ax=b, to solve x
    // lookup table
    void* d_lookup_table_in_;
    void* d_lookup_table_out_;

    // TODO: add allocator, reserved for future use
    void* cpu_allocator_ = nullptr;
    void* gpu_allocator_ = nullptr;

    void allocateMemory();
    void freeMemory();

   public:
   
    Qcu(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt,
        QCU_PRECISION outputFloatPrecision,
        QCU_PRECISION iterateFloatPrecision = QCU_DOUBLE_PRECISION,
        int nColors = 3, int mInputs = 1, double mass = 0.0,
        bool inverterEnabled = false)
        : inverterEnabled_(inverterEnabled),
          nColors_(nColors),
          mInput_(mInputs),
          mass_(mass),
          kappa_(1.0 / (2.0 * (4.0 + mass))),
          lattDesc_(Lx, Ly, Lz, Lt),
          procDesc_(Gx, Gy, Gz, Gt),
          outputFloatPrecision_(outputFloatPrecision),
          dslashParam_(nullptr),
          dslash_(nullptr),
          gauge_(nullptr),
          fp64Gauge_(nullptr),
          fp32Gauge_(nullptr),
          fp16Gauge_(nullptr),
          fermionIn_MRHS_(nullptr),
          fermionOut_MRHS_(nullptr),
          iterateFloatPrecision_(iterateFloatPrecision)
    {
        allocateMemory();
    }

    ~Qcu() { freeMemory(); }

    QcuLattDesc lattDesc() const { return lattDesc_; }
    QcuProcDesc procDesc() const { return procDesc_; }
    
    int32_t color() const { return nColors_; }
    int32_t rhs_num () const { return mInput_; }
    int32_t nSpin () const { return Ns; }

    void getDslash(DSLASH_TYPE dslashType, double mass);
    void startDslash(int parity, bool daggerFlag = false);
    void loadGauge(void *gauge, QCU_PRECISION floatPrecision);

    void pushBackFermions(void *fermionOut, void *fermionIn);
    void setInverterEnabled(bool enabled) { inverterEnabled_ = enabled; }
    // solve Ax = b
    void solveFermions(int max_iteration, double p_max_prec);
};

}  // namespace qcu