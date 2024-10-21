#include <cuda_fp16.h>

#include <cassert>
#include <cstdlib>

#include "../tests/public_complex_vector.h"
#include "data_format/qcu_data_format_shift.cuh"
#include "qcu_interface.h"
#include "qcu_public.h"
#include "qcu_utils.h"          // div_ceil
#include "qcu_wmma_constant.h"  // use this to debug
#include "solver/bicgstab.cuh"
#include "timer/timer.h"

#include "check_error/check_cuda.cuh"

#include "lqcd_read_write.h"
#include "precondition/even_odd_precondition.h"

#include "qcu_blas/qcu_blas.h"

namespace qcu {

void Qcu::allocateMemory() {
    int Lx = lattice_desc_.data[X_DIM];
    int Ly = lattice_desc_.data[Y_DIM];
    int Lz = lattice_desc_.data[Z_DIM];
    int Lt = lattice_desc_.data[T_DIM];

    int vol = Lx * Ly * Lz * Lt;
    int colorSpinorMrhs_size = vol * Ns * n_colors_ * m_input_;  // even and odd
    int gauge_size = Nd * vol * n_colors_ * n_colors_;   // even and odd

    switch (compute_floatprecision_) {
        case QCU_HALF_PRECISION: {
            CHECK_CUDA(cudaMalloc(&fermion_in_mrhs_, 2 * colorSpinorMrhs_size * sizeof(half)));
            CHECK_CUDA(cudaMalloc(&fermion_out_mrhs_, 2 * colorSpinorMrhs_size * sizeof(half)));
        } break;
        case QCU_SINGLE_PRECISION: {
            CHECK_CUDA(cudaMalloc(&fermion_in_mrhs_, 2 * colorSpinorMrhs_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&fermion_out_mrhs_, 2 * colorSpinorMrhs_size * sizeof(float)));
        } break;
        case QCU_DOUBLE_PRECISION: {
            CHECK_CUDA(cudaMalloc(&fermion_in_mrhs_, 2 * colorSpinorMrhs_size * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&fermion_out_mrhs_, 2 * colorSpinorMrhs_size * sizeof(double)));
        } break;

        default:
            break;
    }
    // gauge field
    CHECK_CUDA(cudaMalloc(&fp64_gauge_, 2 * gauge_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&fp32_gauge_, 2 * gauge_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&fp16_gauge_, 2 * gauge_size * sizeof(half)));

    CHECK_CUDA(cudaMalloc(&d_lookup_table_in_, sizeof(void*) * m_input_));
    CHECK_CUDA(cudaMalloc(&d_lookup_table_out_, sizeof(void*) * m_input_));
}

void Qcu::freeMemory() {
    if (dslash_param_ != nullptr) {
        delete dslash_param_;
    }
    if (dslash_ != nullptr) {
        delete dslash_;
    }

    if (fp64_gauge_ != nullptr) {
        CHECK_CUDA(cudaFree(fp64_gauge_));
    }
    if (fp32_gauge_ != nullptr) {
        CHECK_CUDA(cudaFree(fp32_gauge_));
    }
    if (fp16_gauge_ != nullptr) {
        CHECK_CUDA(cudaFree(fp16_gauge_));
    }
    if (fermion_in_mrhs_ != nullptr) {
        CHECK_CUDA(cudaFree(fermion_in_mrhs_));
    }
    if (fermion_out_mrhs_ != nullptr) {
        CHECK_CUDA(cudaFree(fermion_out_mrhs_));
    }

    if (d_lookup_table_in_ != nullptr) {
        CHECK_CUDA(cudaFree(d_lookup_table_in_));
    }

    if (d_lookup_table_out_ != nullptr) {
        CHECK_CUDA(cudaFree(d_lookup_table_out_));
    }
}

void Qcu::get_dslash(DSLASH_TYPE dslashType, double mass) {
    if (nullptr != dslash_) {
        delete dslash_;
    }
    void* gauge;
    switch (compute_floatprecision_) {
        case QCU_HALF_PRECISION:
            gauge = fp16_gauge_;
            break;
        case QCU_SINGLE_PRECISION:
            gauge = fp32_gauge_;
            break;
        case QCU_DOUBLE_PRECISION:
            gauge = fp64_gauge_;
            break;
        default:
            errorQcu("Unsupported float precision\n");
    }

    bool default_dagger_flag = false;
    mass_ = mass;
    kappa_ = (1.0 / (2.0 * (4.0 + mass)));

    dslash_param_ = new DslashParam
                    (
                        default_dagger_flag, compute_floatprecision_, n_colors_, m_input_,
                        QCU_PARITY::EVEN_PARITY, kappa_, fermion_in_mrhs_, fermion_out_mrhs_,
                        gauge, &lattice_desc_, &process_desc_
                    );

    switch (dslashType) {
        case DSLASH_TYPE::DSLASH_WILSON:
            dslash_ = new WilsonDslash(dslash_param_);
            break;

        default: {
          errorQcu("Unsupported dslash type\n");
          break;
        }

    }
}

void Qcu::start_dslash(int parity, bool daggerFlag) {
    if (nullptr == dslash_) {
        errorQcu("Dslash is not initialized\n");
    }
    if (fermion_in_queue_.size() != m_input_ || fermion_out_vec_.size() != m_input_) {
        errorQcu("Fermion queue is not full\n");
    }
// #define DEBUG_INTERFACE
#ifdef DEBUG_INTERFACE
    std::cout << "DEBUG_INFO: fermionInQueue:";
    std::cout << "fermion.size() = " << fermion_in_queue_.size() << std::endl;
    for (int i = 0; i < fermion_in_queue_.size(); ++i) {
        std::cout << fermion_in_queue_[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "DEBUG_INFO: fermionOutQueue:";
    for (int i = 0; i < fermion_out_vec_.size(); ++i) {
        std::cout << fermion_out_vec_[i] << " ";
    }
    std::cout << std::endl;
#endif
    const int Lx = lattice_desc_.data[X_DIM];
    const int Ly = lattice_desc_.data[Y_DIM];
    const int Lz = lattice_desc_.data[Z_DIM];
    const int Lt = lattice_desc_.data[T_DIM];
    dslash_param_->parity = parity;
    dslash_param_->daggerFlag = daggerFlag;

    dslash_param_->fermionIn_MRHS = fermion_in_mrhs_;
    dslash_param_->fermionOut_MRHS = fermion_out_mrhs_;

    CHECK_CUDA(cudaMemcpy(d_lookup_table_in_, fermion_in_queue_.data(), sizeof(void*) * fermion_in_queue_.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lookup_table_out_, fermion_out_vec_.data(), sizeof(void*) * fermion_in_queue_.size(),
                          cudaMemcpyHostToDevice));

    TIMER_EVENT(colorSpinorGather(fermion_in_mrhs_, compute_floatprecision_, d_lookup_table_in_,
                                out_float_precision_, Lx, Ly, Lz, Lt,
                      n_colors_, m_input_, NULL), 0, "gather");
    CHECK_CUDA(cudaDeviceSynchronize());

    // real op
    int mv_flops = (8 * n_colors_ - 2) * n_colors_; // (8 * in.Ncolor() - 2) * in.Ncolor();
    int num_mv = Ns / 2;
    double num_op = Lx * Ly * Lz * Lt / 2 * m_input_ * (
        2 * Nd * Ns * n_colors_ +
        2 * Nd * num_mv * mv_flops +
        (2 * Nd - 1) * Ns * n_colors_
    );

    [[maybe_unused]] double real_num_op = 0;
    {
        using namespace device;
        int wmma_m = 8;
        int wmma_n = 8;
        int wmma_k = 4;
        int warp_line = div_ceil(n_colors_, wmma_m);
        int warp_col = div_ceil(m_input_, wmma_n);
        
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
    TIMER_EVENT(colorSpinorScatter(d_lookup_table_out_, out_float_precision_, fermion_out_mrhs_,
                              compute_floatprecision_, Lx, Ly, Lz,
                              Lt, n_colors_, m_input_, NULL), 0, "scatter");
    CHECK_CUDA(cudaDeviceSynchronize());

    fermion_in_queue_.clear();
    fermion_out_vec_.clear();
}

void Qcu::mat_qcu (bool daggerFlag) {
    if (nullptr == dslash_) {
        errorQcu("Dslash is not initialized\n");
    }
    if (fermion_in_queue_.size() != m_input_ || fermion_out_vec_.size() != m_input_) {
        errorQcu("Fermion queue is not full\n");
    }

    const int Lx = lattice_desc_.data[X_DIM];
    const int Ly = lattice_desc_.data[Y_DIM];
    const int Lz = lattice_desc_.data[Z_DIM];
    const int Lt = lattice_desc_.data[T_DIM];
    // dslash_param_->parity = parity;
    dslash_param_->daggerFlag = daggerFlag;
    dslash_param_->fermionIn_MRHS = fermion_in_mrhs_;
    dslash_param_->fermionOut_MRHS = fermion_out_mrhs_;

    Complex host_kappa = Complex<OutputFloat>(kappa_, 0);
    CHECK_CUDA(cudaMalloc(&device_kappa_, sizeof(Complex<OutputFloat>) ));
    CHECK_CUDA(cudaMemcpy(device_kappa_, &host_kappa, sizeof(Complex<OutputFloat>), cudaMemcpyHostToDevice));

    vector<void*> fermion_in_half (fermion_in_queue_.size());
    vector<void*> fermion_out_half (fermion_out_vec_.size());
    const int vol = Lx * Ly * Lz * Lt;
    const int fermion_half_len = (vol / 2) * Ns * n_colors_ * m_input_;
    // mat_qcu = fermionIn - kappa fermionOut   
    qcu::qcu_blas::Complex_xsay<OutputFloat> xsay_op;

    for (int parity =0; parity < 2; ++parity) {
        dslash_param_->parity = parity;
        for (int i = 0; i < m_input_; ++i) {
            fermion_out_half[i] = static_cast<Complex<OutputFloat>*>(fermion_out_vec_[i]) + parity * fermion_half_len;
            fermion_in_half[i] = static_cast<Complex<OutputFloat>*>(fermion_in_queue_[i]) + (1 - parity) * fermion_half_len;
        }
        CHECK_CUDA(
            cudaMemcpy(d_lookup_table_in_, fermion_in_half.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice)
        );
        CHECK_CUDA(
            cudaMemcpy(d_lookup_table_out_, fermion_out_half.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice)
        );
        colorSpinorGather(fermion_in_mrhs_, compute_floatprecision_, d_lookup_table_in_, out_float_precision_,
            Lx, Ly, Lz, Lt, n_colors_, m_input_, NULL);
        CHECK_CUDA(cudaDeviceSynchronize());

        dslash_->apply();
        
        CHECK_CUDA(cudaDeviceSynchronize());

        // qcu::qcu_blas::Complex_xsay<OutputFloat>::Complex_xsayArgument arg (
        //     static_cast<Complex<OutputFloat>*>(fermion_out_mrhs_),   // Complex<_Float>* res,
        //     static_cast<Complex<OutputFloat>*>(fermion_in_mrhs_),    // Complex<_Float>* x,
        //     static_cast<Complex<OutputFloat>*>(device_kappa_),      // Complex<_Float>* a,
        //     static_cast<Complex<OutputFloat>*>(fermion_out_mrhs_),   // Complex<_Float>* y,
        //     fermion_half_len,                                       // int single_vec_len,
        //     1,                                                      // int inc_idx,
        //     nullptr                                                 // cudaStream_t stream = nullptr
        // );
        // xsay_op(arg);

        colorSpinorScatter(d_lookup_table_out_, out_float_precision_, fermion_out_mrhs_, compute_floatprecision_,
            Lx, Ly, Lz, Lt, n_colors_, m_input_, NULL);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(
        cudaMemcpy(d_lookup_table_in_, fermion_in_queue_.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice)
    );
    CHECK_CUDA(
        cudaMemcpy(d_lookup_table_out_, fermion_out_vec_.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice)
    );
    colorSpinorGather(fermion_in_mrhs_, compute_floatprecision_, d_lookup_table_in_, out_float_precision_,
            Lx * 2, Ly, Lz, Lt, n_colors_, m_input_, NULL);
    colorSpinorGather(fermion_out_mrhs_, compute_floatprecision_, d_lookup_table_out_, out_float_precision_,
            Lx * 2, Ly, Lz, Lt, n_colors_, m_input_, NULL);
    qcu::qcu_blas::Complex_xsay<OutputFloat>::Complex_xsayArgument arg (
        static_cast<Complex<OutputFloat>*>(fermion_out_mrhs_),   // Complex<_Float>* res,
        static_cast<Complex<OutputFloat>*>(fermion_in_mrhs_),    // Complex<_Float>* x,
        static_cast<Complex<OutputFloat>*>(device_kappa_),      // Complex<_Float>* a,
        static_cast<Complex<OutputFloat>*>(fermion_out_mrhs_),   // Complex<_Float>* y,
        fermion_half_len * 2,                                       // int single_vec_len,
        1,                                                      // int inc_idx,
        nullptr                                                 // cudaStream_t stream = nullptr
    );
    xsay_op(arg);
    colorSpinorScatter(d_lookup_table_out_, out_float_precision_, fermion_out_mrhs_, compute_floatprecision_,
            Lx * 2, Ly, Lz, Lt, n_colors_, m_input_, NULL);
    
    CHECK_CUDA(cudaFree(device_kappa_));
    fermion_in_queue_.clear();
    fermion_out_vec_.clear();
}
void Qcu::load_gauge(void* gauge, QCU_PRECISION floatPrecision) {
    gauge_external_ = gauge;
    int Lx = lattice_desc_.data[X_DIM];
    int Ly = lattice_desc_.data[Y_DIM];
    int Lz = lattice_desc_.data[Z_DIM];
    int Lt = lattice_desc_.data[T_DIM];
    int complex_vector_length = Nd * Lx * Ly * Lz * Lt * n_colors_ * n_colors_;
    
    assert(floatPrecision == QCU_DOUBLE_PRECISION || floatPrecision == QCU_SINGLE_PRECISION ||
           floatPrecision == QCU_HALF_PRECISION);
    copyComplexVector_interface(fp64_gauge_, QCU_DOUBLE_PRECISION, gauge_external_, floatPrecision, complex_vector_length);
    copyComplexVector_interface(fp32_gauge_, QCU_SINGLE_PRECISION, gauge_external_, floatPrecision, complex_vector_length);
    copyComplexVector_interface(fp16_gauge_, QCU_HALF_PRECISION, gauge_external_, floatPrecision, complex_vector_length);
}

void Qcu::push_back_fermion(void* fermionOut, void* fermionIn) {
    if (fermion_in_queue_.size() >= m_input_ || fermion_out_vec_.size() >= m_input_) {
        errorQcu("Fermion queue is full\n");
    }
    fermion_in_queue_.push_back(fermionIn);
    fermion_out_vec_.push_back(fermionOut);
}



void Qcu::solve_fermions(int max_iteration, double max_precision) {
  const int Lx = lattice_desc_.data[X_DIM];
  const int Ly = lattice_desc_.data[Y_DIM];
  const int Lz = lattice_desc_.data[Z_DIM];
  const int Lt = lattice_desc_.data[T_DIM];
  const int vol = Lx * Ly * Lz * Lt;
  const int colorSpinor_len = Ns * n_colors_;

  if (m_input_ != fermion_in_queue_.size()) {
    errorQcu("number of fermion is different from mInput\n");
  } else {
    printf("numbers matched, now begin bicg\n");
  }

//   void* d_lookup_table_in;
//   void* d_lookup_table_out;
//   CHECK_CUDA(cudaMalloc(&d_lookup_table_in, sizeof(void*) * m_input_));
//   CHECK_CUDA(cudaMalloc(&d_lookup_table_out, sizeof(void*) * m_input_));

  vector<void*> fermionIn_queue_odd(fermion_in_queue_.size());
  vector<void*> fermionOut_queue_odd(fermion_out_vec_.size());
  for (int i = 0; i < fermion_in_queue_.size(); i++) {
    fermionIn_queue_odd[i] = static_cast<Complex<OutputFloat>*>(fermion_in_queue_[i]) + colorSpinor_len * vol / 2;
    fermionOut_queue_odd[i] = static_cast<Complex<OutputFloat>*>(fermion_out_vec_[i]) + colorSpinor_len * vol / 2;
  }

  void* fermionIn_MRHS_even = fermion_in_mrhs_;
  void* fermionIn_MRHS_odd = static_cast<Complex<OutputFloat>*>(fermion_in_mrhs_)
                                + colorSpinor_len * m_input_ * vol / 2;
  // gather even
  CHECK_CUDA(cudaMemcpy(d_lookup_table_in_,  fermion_in_queue_.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice));
  TIMER_EVENT(colorSpinorGather(fermionIn_MRHS_even, compute_floatprecision_,
                                d_lookup_table_in_,   out_float_precision_,
                                Lx, Ly, Lz, Lt, n_colors_, m_input_, NULL)
      , 0, "gather");

  // gather odd
  CHECK_CUDA(cudaMemcpy(d_lookup_table_in_, fermionIn_queue_odd.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice));
  TIMER_EVENT(colorSpinorGather(fermionIn_MRHS_odd, compute_floatprecision_,
                                d_lookup_table_in_, out_float_precision_,
                                Lx, Ly, Lz, Lt, n_colors_, m_input_, NULL)
      , 0, "gather");



  // SOLVE
  void* gauge;
  if (out_float_precision_ == QCU_DOUBLE_PRECISION) {
    gauge = fp64_gauge_;
  }
  else if (out_float_precision_ == QCU_SINGLE_PRECISION) {
    gauge = fp32_gauge_;
  }
  else {
    gauge = fp16_gauge_;
  }
  qcu::solver::BiCGStabParam param{
    .nColor         = n_colors_,
    .mInput         = m_input_,
    .kappa          = kappa_,
    .output_x_mrhs  = fermion_out_mrhs_,
    .input_b_mrhs   = fermion_in_mrhs_,
    .gauge          = gauge,
    .lattDesc       = &lattice_desc_,
    .procDesc       = &process_desc_,
    .stream1        = nullptr,
    .stream2        = nullptr
  };
  solver::ApplyBicgStab(param, out_float_precision_, compute_floatprecision_,
                            max_iteration, max_precision);

  // scatter
  void* fermionOut_MRHS_even = fermion_out_mrhs_;
  void* fermionOut_MRHS_odd = static_cast<Complex<OutputFloat>*>(fermion_out_mrhs_)
                                  + colorSpinor_len * m_input_ * vol / 2;
  // scatter even
  CHECK_CUDA(cudaMemcpy(d_lookup_table_out_, fermion_out_vec_.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice));
  TIMER_EVENT(
    colorSpinorScatter( d_lookup_table_out_,   out_float_precision_,
                        fermionOut_MRHS_even, compute_floatprecision_,
                        Lx, Ly, Lz, Lt, n_colors_, m_input_, NULL),
    0, "scatter");
  // scatter odd
  CHECK_CUDA(cudaMemcpy(d_lookup_table_out_, fermionOut_queue_odd.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice));
  TIMER_EVENT(
    colorSpinorScatter( d_lookup_table_out_,  out_float_precision_,
                        fermionOut_MRHS_odd, compute_floatprecision_,
                        Lx, Ly, Lz, Lt, n_colors_, m_input_, NULL),
    0, "scatter");
  CHECK_CUDA(cudaStreamSynchronize(NULL));
  // free lookup-table
//   CHECK_CUDA(cudaFree(d_lookup_table_in));
//   CHECK_CUDA(cudaFree(d_lookup_table_out));
  fermion_in_queue_.clear();
  fermion_out_vec_.clear();
}

void Qcu::read_gauge_from_file (const char* file_path, void* data_ptr) {
    std::string file = file_path;
    QcuHeader qcuHeader;
    MPI_Desc mpi_desc;
    Latt_Desc latt_desc;

#pragma unroll 
    for (int i = 0; i < Nd; ++i) {
        mpi_desc.data[i] = 1;   // 
        latt_desc.data[i] = lattice_desc_.data[i];
    }
    // MPI_Coordinate: todo 
    MPI_Coordinate coord;
#pragma unroll
    for (int i = 0; i < Nd; ++i) {
        coord.data[i] = 0;
    }

    GaugeReader gaugeReader(file, qcuHeader, mpi_desc, coord, latt_desc);

    auto gauge_length = qcuHeader.GaugeLength();
    // std::cout << gauge_length << std::endl; 
    Complex<double>* host_ptr = new Complex<double>[gauge_length];
    gaugeReader.read_gauge(reinterpret_cast<std::complex<double>*>(host_ptr), 0);
    Complex<double>* unpreconditioned;
    CHECK_CUDA (cudaMalloc(&unpreconditioned, sizeof(Complex<double>) * gauge_length));
    CHECK_CUDA(cudaMemcpy(unpreconditioned, host_ptr, sizeof(Complex<double>) * gauge_length, cudaMemcpyHostToDevice));
    qcu::GaugeEOPreconditioner<double> preconditioner;
    preconditioner.reverse(static_cast<Complex<double>*>(data_ptr), 
                            unpreconditioned, 
                            latt_desc, 
                            qcuHeader.GaugeSiteLength(),
                            4,  
                            nullptr);

    CHECK_CUDA(cudaFree(unpreconditioned));
    delete[] host_ptr;
}

}  // namespace qcu
