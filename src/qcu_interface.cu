#include <cuda_fp16.h>

#include <cassert>
#include <cstdlib>

#include "../tests/public_complex_vector.h"
#include "data_format/qcu_data_format_shift.cuh"
#include "qcu_interface.h"
#include "qcu_macro.h"
#include "qcu_utils.h"          // div_ceil
#include "qcu_wmma_constant.h"  // use this to debug
#include "solver/bicgstab.cuh"
#include "timer/timer.h"
namespace qcu {
// #define DEBUG_FUNC
void Qcu::allocateMemory() {
    int Lx = lattDesc_.dims[X_DIM];
    int Ly = lattDesc_.dims[Y_DIM];
    int Lz = lattDesc_.dims[Z_DIM];
    int Lt = lattDesc_.dims[T_DIM];

    int vol = Lx * Ly * Lz * Lt;
    int colorSpinorMrhs_size = vol * Ns * nColors_ * mInput_;  // even and odd
    int gauge_size = Nd * vol * nColors_ * nColors_;   // even and odd

    switch (iterateFloatPrecision_) {
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
    switch (iterateFloatPrecision_) {
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
                        default_dagger_flag, iterateFloatPrecision_, nColors_, mInput_,
                        QCU_PARITY::EVEN_PARITY, kappa_, fermionIn_MRHS_, fermionOut_MRHS_,
                        gauge, &lattDesc_, &procDesc_
                    );

    switch (dslashType) {
        case DSLASH_TYPE::DSLASH_WILSON:
            dslash_ = new WilsonDslash(dslashParam_);
            break;

        default: {
          errorQcu("Unsupported dslash type\n");
          break;
        }

    }
}

void Qcu::startDslash(int parity, bool daggerFlag) {
    if (nullptr == dslash_) {
        errorQcu("Dslash is not initialized\n");
    }
    if (fermionIn_queue_.size() != mInput_ || fermionOut_queue_.size() != mInput_) {
        errorQcu("Fermion queue is not full\n");
    }
    const int Lx = lattDesc_.dims[X_DIM];
    const int Ly = lattDesc_.dims[Y_DIM];
    const int Lz = lattDesc_.dims[Z_DIM];
    const int Lt = lattDesc_.dims[T_DIM];
    dslashParam_->parity = parity;
    dslashParam_->daggerFlag = daggerFlag;

    dslashParam_->fermionIn_MRHS = fermionIn_MRHS_;
    dslashParam_->fermionOut_MRHS = fermionOut_MRHS_;

    // lookup table
    void* d_lookup_table_in;
    void* d_lookup_table_out;
    CHECK_CUDA(cudaMalloc(&d_lookup_table_in, sizeof(void*) * fermionIn_queue_.size()));    // TODO : fermionIn_queue_.size()改为mInput
    CHECK_CUDA(cudaMalloc(&d_lookup_table_out, sizeof(void*) * fermionOut_queue_.size()));

    CHECK_CUDA(cudaMemcpy(d_lookup_table_in, fermionIn_queue_.data(), sizeof(void*) * fermionIn_queue_.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lookup_table_out, fermionOut_queue_.data(), sizeof(void*) * fermionIn_queue_.size(),
                          cudaMemcpyHostToDevice));

    TIMER_EVENT(colorSpinorGather(fermionIn_MRHS_, iterateFloatPrecision_, d_lookup_table_in,
                                outputFloatPrecision_, Lx, Ly, Lz, Lt,
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
    }
    
    TIMER_EVENT(dslash_->apply(), num_op, "wilson dslash");
    // TIMER_EVENT(dslash_->apply(), real_num_op, "wilson dslash real op");

    TIMER_EVENT(colorSpinorScatter(d_lookup_table_out, outputFloatPrecision_, fermionOut_MRHS_,
                              iterateFloatPrecision_, Lx, Ly, Lz,
                              Lt, nColors_, mInput_, NULL), 0, "scatter");
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
#ifdef DEBUG_FUNC
    cout << "[[[[[[[[[[[[[[[[[[[ pushback " << fermionOut  << endl;
#endif
}


#ifdef DEBUG_FUNC
void Qcu::solveFermions(int max_iteration, double max_precision) {
  const int Lx = lattDesc_.dims[X_DIM];
  const int Ly = lattDesc_.dims[Y_DIM];
  const int Lz = lattDesc_.dims[Z_DIM];
  const int Lt = lattDesc_.dims[T_DIM];
  const int vol = Lx * Ly * Lz * Lt;
  const int colorSpinor_len = Ns * nColors_;

  if (mInput_ != fermionIn_queue_.size() || mInput_ != fermionOut_queue_.size()) {
    errorQcu("number of fermion is different from mInput\n");
  } else {
    printf("numbers matched, now begin bicg\n");
  }

  void* d_lookup_table_in;
  void* d_lookup_table_out;
  CHECK_CUDA(cudaMalloc(&d_lookup_table_in, sizeof(void*) * mInput_));
  CHECK_CUDA(cudaMalloc(&d_lookup_table_out, sizeof(void*) * mInput_));

  // vector<void*> fermionIn_queue_odd(fermionIn_queue_.size());
  // vector<void*> fermionOut_queue_odd(fermionOut_queue_.size());
  vector<void*> fermionIn_queue_odd;//(fermionIn_queue_.size());
  vector<void*> fermionOut_queue_odd;
  for (int i = 0; i < fermionIn_queue_.size(); i++) {
    // fermionIn_queue_odd[i]  = static_cast<Complex<OutputFloat>*>(fermionIn_queue_[i])  + colorSpinor_len * mInput_ * vol / 2;
    // fermionOut_queue_odd[i] = static_cast<Complex<OutputFloat>*>(fermionOut_queue_[i]) + colorSpinor_len * mInput_ * vol / 2;
    fermionIn_queue_odd.push_back(static_cast<Complex<OutputFloat>*>(fermionIn_queue_[i])  + colorSpinor_len  * vol / 2);
    fermionOut_queue_odd.push_back(static_cast<Complex<OutputFloat>*>(fermionOut_queue_[i]) + colorSpinor_len * vol / 2);
  }
  cout << "fermionOut ptrs================================="<< endl;
  for (int i = 0; i < fermionOut_queue_.size(); i++) {
    cout << "fermionOut[" << i << "] = " << fermionOut_queue_[i] << endl;
  }

  cout << "DEBUG ======================" << endl;
  cout << "FermionOut even size: " << fermionOut_queue_.size() << endl;
  cout << "FermionOut odd size: " << fermionIn_queue_odd.size() << endl;
  for (int i = 0; i < fermionOut_queue_.size(); i++) {
    cout << "Even ptr " << fermionOut_queue_[i] << ", odd ptr " << fermionIn_queue_odd[i] << endl;
  }
  cout << "DEBUG =======================" << endl;

  void* fermionIn_MRHS_even = fermionIn_MRHS_;
  void* fermionIn_MRHS_odd = static_cast<Complex<OutputFloat>*>(fermionIn_MRHS_)
                                + colorSpinor_len * mInput_ * vol / 2;
  // gather even
  CHECK_CUDA(cudaMemcpy(d_lookup_table_in,  fermionIn_queue_.data(), sizeof(void*) * mInput_, cudaMemcpyHostToDevice));
  TIMER_EVENT(colorSpinorGather(fermionIn_MRHS_even, iterateFloatPrecision_,
                                d_lookup_table_in,   outputFloatPrecision_,
                                Lx, Ly, Lz, Lt, nColors_, mInput_, NULL)
      , 0, "gather");

  // gather odd
  CHECK_CUDA(cudaMemcpy(d_lookup_table_in, fermionIn_queue_odd.data(), sizeof(void*) * mInput_, cudaMemcpyHostToDevice));
  TIMER_EVENT(colorSpinorGather(fermionIn_MRHS_odd, iterateFloatPrecision_,
                                d_lookup_table_in, outputFloatPrecision_,
                                Lx, Ly, Lz, Lt, nColors_, mInput_, NULL)
      , 0, "gather");

  // 原地复制
  CHECK_CUDA(cudaMemcpy(fermionOut_MRHS_, fermionIn_MRHS_,
             sizeof(Complex<OutputFloat>) * vol * colorSpinor_len * mInput_, cudaMemcpyDeviceToDevice));

  // scatter
  void* fermionOut_MRHS_even = fermionOut_MRHS_;
  void* fermionOut_MRHS_odd = static_cast<Complex<OutputFloat>*>(fermionOut_MRHS_)
                                  + colorSpinor_len * mInput_ * vol / 2;
  // scatter even
  CHECK_CUDA(cudaMemcpy(d_lookup_table_out, fermionOut_queue_.data(), sizeof(void*) * mInput_, cudaMemcpyHostToDevice));
  TIMER_EVENT(
    colorSpinorScatter( d_lookup_table_out,   outputFloatPrecision_,
                        fermionOut_MRHS_even, iterateFloatPrecision_,
                        Lx, Ly, Lz, Lt, nColors_, mInput_, NULL),
    0, "scatter");
  // scatter odd
  CHECK_CUDA(cudaMemcpy(d_lookup_table_out, fermionOut_queue_odd.data(), sizeof(void*) * mInput_, cudaMemcpyHostToDevice));
  TIMER_EVENT(
    colorSpinorScatter( d_lookup_table_out,  outputFloatPrecision_,
                        fermionOut_MRHS_odd, iterateFloatPrecision_,
                        Lx, Ly, Lz, Lt, nColors_, mInput_, NULL),
    0, "scatter");
  CHECK_CUDA(cudaStreamSynchronize(NULL));


  // DEBUG
  Complex<double> *fermionOut_HOST = new Complex<double>[Ns * nColors_];
  cout << "fermionOut_MRHS_even: " << fermionOut_MRHS_even << endl;
  cout << "fermionOut_MRHS_odd: " << fermionOut_MRHS_odd << endl;
  cout << "COPY RES=================================== " << endl;

  cout << "ODD[0] first 12" << endl;
  CHECK_CUDA(cudaMemcpy(fermionOut_HOST, fermionIn_queue_odd[0], sizeof(Complex<double>) * Ns * nColors_, cudaMemcpyDeviceToHost));
  cout << "######################################" << endl;
  for (int i = 0; i < Ns * nColors_; i++) {
    cout << "[" << i << "] = " << fermionOut_HOST[i].real() << ", " << fermionOut_HOST[i].imag() << endl;
  }
  cout << "######################################" << endl;
  // cout << "EVEN first 12" << endl;
  // CHECK_CUDA(cudaMemcpy(fermionOut_HOST, fermionOut_MRHS_even, sizeof(Complex<double>) * Ns * nColors_, cudaMemcpyDeviceToHost));
  // cout << "######################################" << endl;
  // for (int i = 0; i < Ns * nColors_; i++) {
  //   cout << "[" << i << "] = " << fermionOut_HOST[i].real() << ", " << fermionOut_HOST[i].imag() << endl;
  // }
  // cout << "######################################" << endl;
  // cout << "ODD first 12 " << endl;
  // CHECK_CUDA(cudaMemcpy(fermionOut_HOST, fermionOut_MRHS_odd, sizeof(Complex<double>) * Ns * nColors_, cudaMemcpyDeviceToHost));
  // cout << "######################################" << endl;
  // for (int i = 0; i < Ns * nColors_; i++) {
  //   cout << "[" << i << "] = " << fermionOut_HOST[i].real() << ", " << fermionOut_HOST[i].imag() << endl;
  // }
  // cout << "######################################" << endl;
  // cout << "[1] ODD first 12 " << endl;
  // CHECK_CUDA(cudaMemcpy(fermionOut_HOST, fermionOut_queue_odd[1], sizeof(Complex<double>) * Ns * nColors_, cudaMemcpyDeviceToHost));
  // cout << "######################################" << endl;
  // for (int i = 0; i < Ns * nColors_; i++) {
  //   cout << "[" << i << "] = " << fermionOut_HOST[i].real() << ", " << fermionOut_HOST[i].imag() << endl;
  // }
  // cout << "######################################" << endl;
  delete[] fermionOut_HOST;

  // free lookup-table
  CHECK_CUDA(cudaFree(d_lookup_table_in));
  CHECK_CUDA(cudaFree(d_lookup_table_out));
  fermionIn_queue_.clear();
  fermionOut_queue_.clear();
}
#else
void Qcu::solveFermions(int max_iteration, double max_precision) {
  const int Lx = lattDesc_.dims[X_DIM];
  const int Ly = lattDesc_.dims[Y_DIM];
  const int Lz = lattDesc_.dims[Z_DIM];
  const int Lt = lattDesc_.dims[T_DIM];
  const int vol = Lx * Ly * Lz * Lt;
  const int colorSpinor_len = Ns * nColors_;

  if (mInput_ != fermionIn_queue_.size()) {
    errorQcu("number of fermion is different from mInput\n");
  } else {
    printf("numbers matched, now begin bicg\n");
  }

  void* d_lookup_table_in;
  void* d_lookup_table_out;
  CHECK_CUDA(cudaMalloc(&d_lookup_table_in, sizeof(void*) * mInput_));
  CHECK_CUDA(cudaMalloc(&d_lookup_table_out, sizeof(void*) * mInput_));

  vector<void*> fermionIn_queue_odd(fermionIn_queue_.size());
  vector<void*> fermionOut_queue_odd(fermionOut_queue_.size());
  for (int i = 0; i < fermionIn_queue_.size(); i++) {
    fermionIn_queue_odd[i] = static_cast<Complex<OutputFloat>*>(fermionIn_queue_[i]) + colorSpinor_len * vol / 2;
    fermionOut_queue_odd[i] = static_cast<Complex<OutputFloat>*>(fermionOut_queue_[i]) + colorSpinor_len * vol / 2;
  }

  cout << "DEBUG ======================" << endl;
  cout << "FermionOut even size: " << fermionOut_queue_.size() << endl;
  cout << "FermionOut odd size: " << fermionIn_queue_odd.size() << endl;
  for (int i = 0; i < fermionOut_queue_.size(); i++) {
    cout << "Even ptr " << fermionOut_queue_[i] << ", odd ptr " << fermionIn_queue_odd[i] << endl;
  }
  cout << "DEBUG =======================" << endl;

  void* fermionIn_MRHS_even = fermionIn_MRHS_;
  void* fermionIn_MRHS_odd = static_cast<Complex<OutputFloat>*>(fermionIn_MRHS_)
                                + colorSpinor_len * mInput_ * vol / 2;
  // gather even
  CHECK_CUDA(cudaMemcpy(d_lookup_table_in,  fermionIn_queue_.data(), sizeof(void*) * mInput_, cudaMemcpyHostToDevice));
  TIMER_EVENT(colorSpinorGather(fermionIn_MRHS_even, iterateFloatPrecision_,
                                d_lookup_table_in,   outputFloatPrecision_,
                                Lx, Ly, Lz, Lt, nColors_, mInput_, NULL)
      , 0, "gather");

  // gather odd
  CHECK_CUDA(cudaMemcpy(d_lookup_table_in, fermionIn_queue_odd.data(), sizeof(void*) * mInput_, cudaMemcpyHostToDevice));
  TIMER_EVENT(colorSpinorGather(fermionIn_MRHS_odd, iterateFloatPrecision_,
                                d_lookup_table_in, outputFloatPrecision_,
                                Lx, Ly, Lz, Lt, nColors_, mInput_, NULL)
      , 0, "gather");



  // SOLVE
  void* gauge;
  if (outputFloatPrecision_ == QCU_DOUBLE_PRECISION) {
    gauge = fp64Gauge_;
  }
  else if (outputFloatPrecision_ == QCU_SINGLE_PRECISION) {
    gauge = fp32Gauge_;
  }
  else {
    gauge = fp16Gauge_;
  }
  qcu::solver::BiCGStabParam param{
    .nColor         = nColors_,
    .mInput         = mInput_,
    .kappa          = kappa_,
    .output_x_mrhs  = fermionOut_MRHS_,
    .input_b_mrhs   = fermionIn_MRHS_,
    .gauge          = gauge,
    .lattDesc       = &lattDesc_,
    .procDesc       = &procDesc_,
    .stream1        = nullptr,
    .stream2        = nullptr
  };
  solver::ApplyBicgStab(param, outputFloatPrecision_, iterateFloatPrecision_,
                            max_iteration, max_precision);

  // scatter
  void* fermionOut_MRHS_even = fermionOut_MRHS_;
  void* fermionOut_MRHS_odd = static_cast<Complex<OutputFloat>*>(fermionOut_MRHS_)
                                  + colorSpinor_len * mInput_ * vol / 2;
  // scatter even
  CHECK_CUDA(cudaMemcpy(d_lookup_table_out, fermionOut_queue_.data(), sizeof(void*) * mInput_, cudaMemcpyHostToDevice));
  TIMER_EVENT(
    colorSpinorScatter( d_lookup_table_out,   outputFloatPrecision_,
                        fermionOut_MRHS_even, iterateFloatPrecision_,
                        Lx, Ly, Lz, Lt, nColors_, mInput_, NULL),
    0, "scatter");
  // scatter odd
  CHECK_CUDA(cudaMemcpy(d_lookup_table_out, fermionOut_queue_odd.data(), sizeof(void*) * mInput_, cudaMemcpyHostToDevice));
  TIMER_EVENT(
    colorSpinorScatter( d_lookup_table_out,  outputFloatPrecision_,
                        fermionOut_MRHS_odd, iterateFloatPrecision_,
                        Lx, Ly, Lz, Lt, nColors_, mInput_, NULL),
    0, "scatter");
  CHECK_CUDA(cudaStreamSynchronize(NULL));

  // free lookup-table
  CHECK_CUDA(cudaFree(d_lookup_table_in));
  CHECK_CUDA(cudaFree(d_lookup_table_out));
  fermionIn_queue_.clear();
  fermionOut_queue_.clear();
}
#endif

}  // namespace qcu
