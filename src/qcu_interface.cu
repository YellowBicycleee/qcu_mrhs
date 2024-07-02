#include <cuda_fp16.h>

#include <cassert>
#include <cstdlib>

#include "data_format/qcu_data_format_shift.cuh"
#include "qcu_interface.h"
#include "qcu_macro.h"
namespace qcu {

void Qcu::allocateMemory() {
    int Lx = lattDesc_.dims[X_DIM];
    int Ly = lattDesc_.dims[Y_DIM];
    int Lz = lattDesc_.dims[Z_DIM];
    int Lt = lattDesc_.dims[T_DIM];

    int vol = Lx * Ly * Lz * Lt;
    int colorSpinorMrhs_size = vol * Ns * nColors_ * mInput_;  // even and odd
    int gauge_size = DIRECTIONS * vol * nColors_ * nColors_;   // even and odd
    switch (dslashFloatPrecision_) {
        case QCU_HALF_PRECISION: {
            CHECK_CUDA(cudaMalloc(&fermionIn_MRHS_, 2 * colorSpinorMrhs_size * sizeof(half)));
            CHECK_CUDA(cudaMalloc(&fermionOut_MRHS_, 2 * colorSpinorMrhs_size * sizeof(half)));
        } break;
        case QCU_SINGLE_PRECISION: {
            CHECK_CUDA(cudaMalloc(&fermionIn_MRHS_, 2 * colorSpinorMrhs_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&fermionOut_MRHS_, 2 * colorSpinorMrhs_size * sizeof(float)));
        } break;
        case QCU_DOUBLE_PRECISION: {
            CHECK_CUDA(cudaMalloc(&fermionIn_MRHS_, 2 * colorSpinorMrhs_size * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&fermionOut_MRHS_, 2 * colorSpinorMrhs_size * sizeof(double)));
        } break;

        default:
            break;
    }
    // gauge field
    //     void *fp64Gauge_;  // double gauge field
    // void *fp32Gauge_;  // single gauge field
    // void *fp16Gauge_;  // half gauge field
    CHECK_CUDA(cudaMalloc(&fp64Gauge_, 2 * gauge_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&fp32Gauge_, 2 * gauge_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&fp16Gauge_, 2 * gauge_size * sizeof(half)));
}

void Qcu::freeMemory() {
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