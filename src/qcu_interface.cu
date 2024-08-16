#include <cuda_fp16.h>

#include <cassert>
#include <cstdlib>

#include "data_format/qcu_data_format_shift.cuh"
#include "qcu_interface.h"
#include "qcu_macro.h"
#include "timer/timer.h"
#include "qcu_wmma_constant.h"  // use this to debug
#include "qcu_utils.h"  // div_ceil
namespace qcu {

void Qcu::allocateMemory() {
    int Lx = lattDesc_.dims[X_DIM];
    int Ly = lattDesc_.dims[Y_DIM];
    int Lz = lattDesc_.dims[Z_DIM];
    int Lt = lattDesc_.dims[T_DIM];

    int vol = Lx * Ly * Lz * Lt;
    int colorSpinorMrhs_size = vol * Ns * nColors_ * mInput_;  // even and odd
    int gauge_size = Nd * vol * nColors_ * nColors_;   // even and odd

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

    dslashParam_ = new DslashParam 
                    (
                        default_dagger_flag, dslashFloatPrecision_, nColors_, mInput_,
                        QCU_PARITY::EVEN_PARITY, kappa_, fermionIn_MRHS_, fermionOut_MRHS_,
                        gauge, &lattDesc_, &procDesc_
                    );

    switch (dslashType) {
        case DSLASH_TYPE::DSLASH_WILSON:
            dslash_ = new WilsonDslash(dslashParam_);
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

    dslashParam_->fermionIn_MRHS = fermionIn_MRHS_;
    dslashParam_->fermionOut_MRHS = fermionOut_MRHS_;
    // DEBUG

    // lookup table
    void* d_lookup_table_in;
    void* d_lookup_table_out;
    CHECK_CUDA(cudaMalloc(&d_lookup_table_in, sizeof(void*) * fermionIn_queue_.size()));    // TODO : fermionIn_queue_.size()改为mInput
    CHECK_CUDA(cudaMalloc(&d_lookup_table_out, sizeof(void*) * fermionOut_queue_.size()));

    CHECK_CUDA(cudaMemcpy(d_lookup_table_in, fermionIn_queue_.data(), sizeof(void*) * fermionIn_queue_.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lookup_table_out, fermionOut_queue_.data(), sizeof(void*) * fermionIn_queue_.size(),
                          cudaMemcpyHostToDevice));

    // colorSpinorGather(fermionIn_MRHS_, dslashFloatPrecision_, d_lookup_table_in, inputFloatPrecision_, Lx, Ly, Lz, Lt,
                    //   nColors_, mInput_, NULL);
    TIMER_EVENT(colorSpinorGather(fermionIn_MRHS_, dslashFloatPrecision_, d_lookup_table_in, inputFloatPrecision_, Lx, Ly, Lz, Lt,
                      nColors_, mInput_, NULL), 0, "gather");
    CHECK_CUDA(cudaDeviceSynchronize());

    // real op
    int mv_flops = (8 * nColors_ - 2) * nColors_; // (8 * in.Ncolor() - 2) * in.Ncolor();
    int num_mv = Ns / 2;
    double num_op = Lx * Ly * Lz * Lt / 2 * mInput_ * (
        2 * Nd * Ns * nColors_ + 
        2 * Nd * num_mv * mv_flops +
        (2 * Nd - 1) * Ns * nColors_
    );

    double real_num_op = 0; 
    {
        using namespace device;
        int wmma_m = 8;
        int wmma_n = 8;
        int wmma_k = 4;
        int warp_line = div_ceil(nColors_, wmma_m);
        int warp_col = div_ceil(mInput_, wmma_n);
        
        int gemm_flops = wmma_m * wmma_n * (8 * wmma_k - 2);

        real_num_op = Lx * Ly * Lz * Lt / 2 * 8 * warp_line * warp_col *(
            // combination
            double(2 * wmma_m * wmma_k * 2) + // 2个矩阵
            // gemm
            double(2 * gemm_flops) +        // 2个gemm
            // add
            double(4 * wmma_m * wmma_n * 2)  // 4个add
        );
        // printf("warp_line = %d, warp_col = %d, gemm_flops = %d, op = %lf\n", warp_line, warp_col, gemm_flops, real_num_op);
        // printf("part1 : %lf, part2 : %lf, part3 : %lf\n", double(2 * wmma_m * wmma_k * 2), double(2 * gemm_flops), double(4 * wmma_m * wmma_n * 2));
    }
    
    TIMER_EVENT(dslash_->apply(), num_op, "wilson dslash");
    // TIMER_EVENT(dslash_->apply(), real_num_op, "wilson dslash real op");

    TIMER_EVENT(colorSpinorScatter(d_lookup_table_out, inputFloatPrecision_, fermionOut_MRHS_, dslashFloatPrecision_, Lx, Ly, Lz,
                       Lt, nColors_, mInput_, NULL), 0, "scatter");
    // colorSpinorScatter(d_lookup_table_out, inputFloatPrecision_, fermionOut_MRHS_, dslashFloatPrecision_, Lx, Ly, Lz,
    //                    Lt, nColors_, mInput_, NULL);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_lookup_table_in));
    CHECK_CUDA(cudaFree(d_lookup_table_out));
    fermionIn_queue_.clear();
    fermionOut_queue_.clear();
}

void Qcu::loadGauge(void* gauge, QCU_PRECISION floatPrecision) {
    gauge_ = gauge;
    int Lx = lattDesc_.dims[X_DIM];
    int Ly = lattDesc_.dims[Y_DIM];
    int Lz = lattDesc_.dims[Z_DIM];
    int Lt = lattDesc_.dims[T_DIM];
    int complex_vector_length = Nd * Lx * Ly * Lz * Lt * nColors_ * nColors_;
    
    assert(floatPrecision == QCU_DOUBLE_PRECISION || floatPrecision == QCU_SINGLE_PRECISION ||
           floatPrecision == QCU_HALF_PRECISION);
    // fp64Gauge_ = gauge;
    // CHECK_CUDA(cudaMemcpy(fp64Gauge_, gauge, sizeof(double) * 2 * complex_vector_length, cudaMemcpyDeviceToDevice));
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
