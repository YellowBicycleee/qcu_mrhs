#include "qcu_interface.cuh"

#include <cassert>
#include <cstdlib>

#include "data_format/qcu_data_format_shift.cuh"
#include "qcu_macro.h"

namespace qcu {

void Qcu::getDslash(DSLASH_TYPE dslashType, double mass) {
    if (nullptr != dslash_) {
        delete dslash_;
    }
    void* gauge;
    switch (dslashFloatPrecision_) {
        case QCU_HALF_PRECISION:
            gauge = fp16Gauge_;
            break;
        case QCU_SINGLE_PRECISION:
            gauge = fp32Gauge_;
            break;
        case QCU_DOUBLE_PRECISION:
            gauge = fp64Gauge_;
            break;
        default:
            errorQcu("Unsupported float precision\n");
    }

    bool default_dagger_flag = false;
    mass_ = mass;
    kappa_ = (1.0 / (2.0 * (4.0 + mass)));

    dslashParam_ = new DslashParam(dslashFloatPrecision_, nColors_, mInput_, kappa_, QCU_PARITY::EVEN_PARITY,
                                   default_dagger_flag, fermionIn_MRHS_, fermionOut_MRHS_, gauge, lattDesc_, procDesc_);

    switch (dslashType) {
        case DSLASH_TYPE::DSLASH_WILSON:
            dslash_ = new WilsonDslash(*dslashParam_);
            break;

        default:
            errorQcu("Unsupported dslash type\n");
            break;
    }
}

void Qcu::startDslash(int parity, bool daggerFlag) {
    if (nullptr == dslash_) {
        errorQcu("Dslash is not initialized\n");
    }
    if (fermionIn_queue_.size() != mInput_ || fermionOut_queue_.size() != mInput_) {
        errorQcu("Fermion queue is not full\n");
    }
    int Lx = lattDesc_.dims[X_DIM];
    int Ly = lattDesc_.dims[Y_DIM];
    int Lz = lattDesc_.dims[Z_DIM];
    int Lt = lattDesc_.dims[T_DIM];
    dslashParam_->parity = parity;
    dslashParam_->daggerFlag = daggerFlag;

    colorSpinorGather(fermionIn_MRHS_, dslashFloatPrecision_, fermionIn_queue_.data(), inputFloatPrecision_, Lx, Ly, Lz,
                      Lt, nColors_, mInput_, NULL);
    dslash_->apply();
    CHECK_CUDA(cudaDeviceSynchronize());

    colorSpinorScatter(fermionOut_queue_.data(), inputFloatPrecision_, fermionOut_MRHS_, dslashFloatPrecision_, Lx, Ly,
                       Lz, Lt, nColors_, mInput_, NULL);

    fermionIn_queue_.clear();
    fermionOut_queue_.clear();
}

void Qcu::loadGauge(void* gauge, QCU_PRECISION floatPrecision) {
    gauge_ = gauge;
    int Lx = lattDesc_.dims[X_DIM];
    int Ly = lattDesc_.dims[Y_DIM];
    int Lz = lattDesc_.dims[Z_DIM];
    int Lt = lattDesc_.dims[T_DIM];
    int complex_vector_length = DIRECTIONS * Lx * Ly * Lz * Lt * nColors_ * nColors_;
    assert(floatPrecision == QCU_DOUBLE_PRECISION || floatPrecision == QCU_SINGLE_PRECISION ||
           floatPrecision == QCU_HALF_PRECISION);
    copyComplexVector_interface(fp64Gauge_, QCU_DOUBLE_PRECISION, gauge_, floatPrecision, complex_vector_length);
    copyComplexVector_interface(fp32Gauge_, QCU_SINGLE_PRECISION, gauge_, floatPrecision, complex_vector_length);
    copyComplexVector_interface(fp16Gauge_, QCU_HALF_PRECISION, gauge_, floatPrecision, complex_vector_length);
}

void Qcu::pushBackFermions(void* fermionOut, void* fermionIn) {
    if (fermionIn_queue_.size() >= mInput_ || fermionOut_queue_.size() >= mInput_) {
        errorQcu("Fermion queue is full\n");
    }
    fermionIn_queue_.push_back(fermionIn);
    fermionOut_queue_.push_back(fermionOut);
}
}  // namespace qcu