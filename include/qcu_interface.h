#pragma once
// #include <cuda_fp16.h>

#include <vector>

#include "desc/qcu_desc.h"
#include "qcd/qcu_dslash.h"
#include "qcu_enum.h"
#include "qcu_macro.h"
namespace qcu {
class Qcu {
    bool inverterEnabled_;
    int nColors_;
    int mInput_;
    double mass_;
    double kappa_;
    QCU_PRECISION inputFloatPrecision_;
    QCU_PRECISION dslashFloatPrecision_;

    QcuLattDesc lattDesc_;
    QcuProcDesc procDesc_;
    DslashParam *dslashParam_;
    Dslash *dslash_;

    std::vector<void *> fermionIn_queue_;
    std::vector<void *> fermionOut_queue_;

    void *gauge_;      // gauge field
    void *fp64Gauge_;  // double gauge field
    void *fp32Gauge_;  // single gauge field
    void *fp16Gauge_;  // half gauge field

    // mrhs fermion field
    void *fermionIn_MRHS_;
    void *fermionOut_MRHS_;

    void allocateMemory();
    void freeMemory() {
        if (dslashParam_ != nullptr) {
            delete dslashParam_;
        }
        if (dslash_ != nullptr) {
            delete dslash_;
        }
        if (gauge_ != nullptr) {
            CHECK_CUDA(cudaFree(gauge_));
        }
        if (fp64Gauge_ != nullptr) {
            CHECK_CUDA(cudaFree(fp64Gauge_));
        }
        if (fp32Gauge_ != nullptr) {
            CHECK_CUDA(cudaFree(fp32Gauge_));
        }
        if (fp16Gauge_ != nullptr) {
            CHECK_CUDA(cudaFree(fp16Gauge_));
        }
        if (fermionIn_MRHS_ != nullptr) {
            CHECK_CUDA(cudaFree(fermionIn_MRHS_));
        }
        if (fermionOut_MRHS_ != nullptr) {
            CHECK_CUDA(cudaFree(fermionOut_MRHS_));
        }
    }

   public:
    Qcu(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt, QCU_PRECISION inputFloatPrecision,
        QCU_PRECISION dslashFloatPrecision = QCU_DOUBLE_PRECISION, int nColors = 3, int mInputs = 1, double mass = 0.0,
        bool inverterEnabled = false)
        : inverterEnabled_(inverterEnabled),
          nColors_(nColors),
          mInput_(mInputs),
          mass_(mass),
          kappa_(1.0 / (2.0 * (4.0 + mass))),
          lattDesc_(Lx, Ly, Lz, Lt),
          procDesc_(Gx, Gy, Gz, Gt),
          inputFloatPrecision_(inputFloatPrecision),
          dslashParam_(nullptr),
          dslash_(nullptr),
          gauge_(nullptr),
          fp64Gauge_(nullptr),
          fp32Gauge_(nullptr),
          fp16Gauge_(nullptr),
          fermionIn_MRHS_(nullptr),
          fermionOut_MRHS_(nullptr),
          dslashFloatPrecision_(dslashFloatPrecision) {
        allocateMemory();
    }

    ~Qcu() { freeMemory(); }

    void getDslash(DSLASH_TYPE dslashType, double mass);
    void startDslash(int parity, bool daggerFlag = false);
    void loadGauge(void *gauge, QCU_PRECISION floatPrecision);

    void pushBackFermions(void *fermionOut, void *fermionIn);
    void setInverterEnabled(bool enabled) { inverterEnabled_ = enabled; }
};

}  // namespace qcu