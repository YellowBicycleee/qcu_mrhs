#pragma once
#include <vector>

#include "desc/qcu_desc.h"
#include "qcu_enum.h"
#include "qcu_macro.h"

namespace qcu {
constexpr int Ndim = 4;

class Qcu {
    int inverterEnabled_;
    int nColors_;
    int mInput_;
    double mass_;
    double kappa_;

    QcuLattDesc<Ndim> lattDesc_;
    QcuProcDesc<Ndim> procDesc_;
    Dirac *dirac_;

    vector<void *> fermionIn_queue_;
    vector<void *> fermionOut_queue_;

    void *gauge_;      // gauge field
    void *fp64Gauge_;  // double gauge field
    void *fp32Gauge_;  // single gauge field
    void *fp16Gauge_;  // half gauge field

    void *fp16fermionIn_;
    void *fp32fermionIn_;
    void *fp64fermionIn_;

    void *fp16fermionOut_;
    void *fp32fermionOut_;
    void *fp64fermionOut_;

   public:
    Qcu(int nColors, int mInputs, double mass, int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt,
        int inverterEnabled = 0)
        : inverterEnabled_(inverterEnabled),
          nColors_(nColors),
          mInput_(mInputs),
          mass_(mass),
          kappa_(1.0 / (2.0 * (4.0 + mass))),
          lattDesc_(Lx, Ly, Lz, Lt),
          procDesc_(Gx, Gy, Gz, Gt),
          dirac_(nullptr),
          gauge_(nullptr),
          fp64Gauge_(nullptr),
          fp32Gauge_(nullptr),
          fp16Gauge_(nullptr),
          fp16fermionIn_(nullptr),
          fp32fermionIn_(nullptr),
          fp64fermionIn_(nullptr),
          fp16fermionOut_(nullptr),
          fp32fermionOut_(nullptr),
          fp64fermionOut_(nullptr) {
        // todo: allocations
    }
    ~Qcu() {
        if (dirac_ != nullptr) {
            delete dirac_;
        }
    }
    // TODO
    void getDslash(DSLASH_TYPE dslashType, double mass, int nColors, int nInputs, QCU_PRECISION floatPrecision,
                   int daggerFlag = 0);
    void startDslash(int parity, int daggerFlag = 0);
    void loadGauge(void *gauge);
    void setInverterEnabled(bool enabled) { inverterEnabled_ = enabled; }
};

}  // namespace qcu