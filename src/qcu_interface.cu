#include <check_error/check_mpi.h>
#include <cuda_fp16.h>
#include <qcu_config/qcu_config.h>

#include <cassert>
#include <cstdlib>

// #include "../tests/public_complex_vector.h"
#include "check_error/check_cuda.cuh"
#include "data_format/fermion.cuh"
#include "data_format/gauge.cuh"
#include "data_format/qcu_data_format_shift.cuh"
#include "io/lqcd_read_write.h"
#include "precondition/even_odd_precondition.h"
#include "qcu_blas/qcu_blas.h"
#include "qcu_interface.h"
#include "qcu_public.h"
#include "qcu_utils.h"          // div_ceil
#include "qcu_wmma_constant.h"  // use this to debug
#include "solver/bicgstab.cuh"
#include "timer/timer.h"
namespace qcu {

void Qcu::allocateMemory() {
    // printf("nspin: %d, ncolor: %d, m_input: %d\n", n_spin_, n_colors_, m_input_);
    assert(n_spin_ > 0);
    int vol = qcu::config::lattice_volume_local();
    int colorSpinorMrhs_size = vol * n_spin_ * n_colors_ * m_input_;  // even and odd
    int gauge_size = Nd * vol * n_colors_ * n_colors_;   // even and odd

    switch (underlying_args_.compute_float_precision) {
        case QcuPrecision::kPrecisionHalf : {
            CHECK_CUDA(cudaMalloc(&fermion_in_mrhs_, 2 * colorSpinorMrhs_size * sizeof(half)));
            CHECK_CUDA(cudaMalloc(&fermion_out_mrhs_, 2 * colorSpinorMrhs_size * sizeof(half)));
        } break;
        case QcuPrecision::kPrecisionSingle : {
            CHECK_CUDA(cudaMalloc(&fermion_in_mrhs_, 2 * colorSpinorMrhs_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&fermion_out_mrhs_, 2 * colorSpinorMrhs_size * sizeof(float)));
        } break;
        case QcuPrecision::kPrecisionDouble: {
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

    if (fp64_gauge_ != nullptr) { CHECK_CUDA(cudaFree(fp64_gauge_)); fp64_gauge_ = nullptr; }
    if (fp32_gauge_ != nullptr) { CHECK_CUDA(cudaFree(fp32_gauge_)); fp32_gauge_ = nullptr; }
    if (fp16_gauge_ != nullptr) { CHECK_CUDA(cudaFree(fp16_gauge_)); fp16_gauge_ = nullptr; }
    if (fermion_in_mrhs_ != nullptr) { CHECK_CUDA(cudaFree(fermion_in_mrhs_)); fermion_in_mrhs_ = nullptr;}
    if (fermion_out_mrhs_ != nullptr) { CHECK_CUDA(cudaFree(fermion_out_mrhs_)); fermion_out_mrhs_ = nullptr; }

    if (d_lookup_table_in_ != nullptr) { CHECK_CUDA(cudaFree(d_lookup_table_in_)); d_lookup_table_in_ = nullptr; }

    if (d_lookup_table_out_ != nullptr) { CHECK_CUDA(cudaFree(d_lookup_table_out_)); d_lookup_table_out_ = nullptr; }
}

void Qcu::get_dslash(DslashType dslashType, double mass, bool anti_periodic_t) {
    anti_periodic_t_ = anti_periodic_t;
    switch (dslashType)
    {
    case DslashType::kDslashWilson:
        n_spin_ = 4;
        break;
    case DslashType::kDslashStaggered:
        n_spin_ = 1;
        break;
    default:
        n_spin_ = -1;
        errorQcu("Unsupported dslash type\n");
        break;
    }
    freeMemory();
    allocateMemory();
    void* gauge;
    switch (underlying_args_.compute_float_precision) {
        case QcuPrecision::kPrecisionHalf:
            gauge = fp16_gauge_;
            break;
        case QcuPrecision::kPrecisionSingle:
            gauge = fp32_gauge_;
            break;
        case QcuPrecision::kPrecisionDouble:
            gauge = fp64_gauge_;
            break;
        default:
            errorQcu("Unsupported float precision\n");
    }

    bool default_dagger_flag = false;
    mass_ = mass;
    kappa_ = (1.0 / (2.0 * (4.0 + mass)));

    std::shared_ptr<qcu::FermionGhost<Nd>> fermion_ghost_ptr 
        = std::make_shared<qcu::FermionGhost<Nd>>(
            underlying_args_.lattice_desc_ptr, 
            config::get_mpi_separated_mask(),
            n_colors_, 
            m_input_, 
            underlying_args_.compute_float_precision);
    dslash_param_ = std::make_shared<DslashParam>
                    (
                        default_dagger_flag, underlying_args_.compute_float_precision, 
                        staggered_phase_, t_boundary_,
                        n_colors_, m_input_,
                        QCU_PARITY::EVEN_PARITY, kappa_, fermion_in_mrhs_, fermion_out_mrhs_,
                        gauge, &(underlying_args_.lattice_desc_ptr), &(underlying_args_.process_desc_ptr),
                        // nullptr, nullptr,
                        config::get_qcu_streams(),
                        fermion_ghost_ptr
                    );

    switch (dslashType) {
        case DslashType::kDslashWilson:
            if (tensor_core_flag_) {
                dslash_ = std::make_shared<qcu::tensorop::WilsonDslash>();
                printf("Use tensor core\n");
            }
            else {
                dslash_ = std::make_shared<qcu::simt::WilsonDslash>();
                printf("Use SIMT\n");
            }
                break;
        case DslashType::kDslashStaggered:
            dslash_ = std::make_shared<qcu::simt::StaggeredDslash>();
            break;
        default: {
            errorQcu("Unsupported dslash type\n");
            break;
        }

    }
}

void Qcu::start_dslash(int parity, bool dagger_flag) {
    if (nullptr == dslash_) {
        errorQcu("Dslash is not initialized\n");
    }


    dslash_param_->staggered_phase = staggered_phase_;
    dslash_param_->parity = parity;
    dslash_param_->dagger_flag = dagger_flag;

    dslash_param_->fermion_in_MRHS = fermion_in_mrhs_;
    dslash_param_->fermion_out_MRHS = fermion_out_mrhs_;


    // begin_gather();
    TIMER_EVENT(dslash_->apply(dslash_param_), dslash_->operations(), "wilson dslash");
    // begin_scatter();
}


template <typename ComputeFloat_, typename ScaleFloat_ = ComputeFloat_>
void Qcu::mat_qcu_template_function (bool dagger_flag) {
    if (nullptr == dslash_) {
        errorQcu("Dslash is not initialized\n");
    }
    if (fermion_in_vec_.size() != m_input_ || fermion_out_vec_.size() != m_input_) {
        errorQcu("Fermion queue is not full\n");
    }

    // dslash_param_->parity = parity;
    dslash_param_->dagger_flag = dagger_flag;
    dslash_param_->fermion_in_MRHS = fermion_in_mrhs_;
    dslash_param_->fermion_out_MRHS = fermion_out_mrhs_;

    Complex<ScaleFloat_> host_kappa = Complex<ScaleFloat_>(kappa_, 0);
    CHECK_CUDA(cudaMalloc(&device_kappa_, sizeof(Complex<ScaleFloat_>) ));
    CHECK_CUDA(cudaMemcpy(device_kappa_, &host_kappa, sizeof(Complex<ScaleFloat_>), cudaMemcpyHostToDevice));

    std::vector<void*> fermion_in_half (m_input_);
    std::vector<void*> fermion_out_half (m_input_);
    const int vol = underlying_args_.lattice_desc_ptr.lattice_volume();
    const int fermion_half_len = (vol / 2) * n_spin_ * n_colors_ * m_input_;
    // mat_qcu = fermionIn - kappa fermionOut   
    qcu::qcu_blas::Complex_xsay<ComputeFloat_, ScaleFloat_> xsay_op;

    for (int parity = 0; parity < 2; ++parity) {
        dslash_param_->parity = parity;
        for (int i = 0; i < m_input_; ++i) {
            fermion_out_half[i] = static_cast<Complex<ComputeFloat_>*>(fermion_out_vec_[i]) + parity * vol / 2 * n_spin_ * n_colors_;
            fermion_in_half[i] = static_cast<Complex<ComputeFloat_>*>(fermion_in_vec_[i]) + (1 - parity) * vol / 2 * n_spin_ * n_colors_;
        }
        CHECK_CUDA(
            cudaMemcpy(d_lookup_table_in_, fermion_in_half.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice)
        );
        CHECK_CUDA(
            cudaMemcpy(d_lookup_table_out_, fermion_out_half.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice)
        );
        colorSpinorGather(fermion_in_mrhs_, underlying_args_.compute_float_precision, d_lookup_table_in_,
                underlying_args_.out_float_precision, *qcu::config::get_lattice_desc_ptr(), n_colors_, m_input_, NULL, n_spin_);
        CHECK_CUDA(cudaDeviceSynchronize());

        dslash_->apply(dslash_param_);
        CHECK_CUDA(cudaDeviceSynchronize());

        colorSpinorScatter(d_lookup_table_out_, underlying_args_.out_float_precision,
            fermion_out_mrhs_, underlying_args_.compute_float_precision,
            *qcu::config::get_lattice_desc_ptr(), n_colors_, m_input_, NULL, n_spin_);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(
        cudaMemcpy(d_lookup_table_in_, fermion_in_vec_.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice)
    );
    CHECK_CUDA(
        cudaMemcpy(d_lookup_table_out_, fermion_out_vec_.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice)
    );
    QcuLattDesc latt_desc_temp = *qcu::config::get_lattice_desc_ptr();
    latt_desc_temp.data[X_DIM] *= 2;

    colorSpinorGather(fermion_in_mrhs_, underlying_args_.compute_float_precision,
        d_lookup_table_in_, underlying_args_.out_float_precision,
            latt_desc_temp, n_colors_, m_input_, nullptr, n_spin_);
    colorSpinorGather(fermion_out_mrhs_, underlying_args_.compute_float_precision,
        d_lookup_table_out_, underlying_args_.out_float_precision,
            latt_desc_temp, n_colors_, m_input_, nullptr, n_spin_);
    typename qcu_blas::Complex_xsay<ComputeFloat_, ScaleFloat_>::Complex_xsayArgument arg (
        static_cast<Complex<ComputeFloat_>*>(fermion_out_mrhs_),   // Complex<_Float>* res,
        static_cast<Complex<ComputeFloat_>*>(fermion_in_mrhs_),    // Complex<_Float>* x,
        static_cast<Complex<ScaleFloat_>*>(device_kappa_),      // Complex<_Float>* a,
        static_cast<Complex<ComputeFloat_>*>(fermion_out_mrhs_),   // Complex<_Float>* y,
        fermion_half_len * 2,                                       // int single_vec_len,
        1,                                                      // int inc_idx,
        nullptr                                                 // cudaStream_t stream = nullptr
    );
    xsay_op(arg);

    colorSpinorScatter(d_lookup_table_out_, underlying_args_.out_float_precision,
        fermion_out_mrhs_, underlying_args_.compute_float_precision,
            latt_desc_temp, n_colors_, m_input_, NULL, n_spin_);
    
    CHECK_CUDA(cudaFree(device_kappa_));
    fermion_in_vec_.clear();
    fermion_out_vec_.clear();
}
void Qcu::mat_qcu (bool dagger_flag) {
    if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionDouble) {
        mat_qcu_template_function<double, double>(dagger_flag);
    } else if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionSingle) {
        mat_qcu_template_function<float, float>(dagger_flag);
    } else if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionHalf) {
        mat_qcu_template_function<half, float>(dagger_flag);
    } else {
        errorQcu("Unsupported float precision\n");
    }
}

void Qcu::load_gauge(void* gauge, QcuPrecision floatPrecision) {
    gauge_external_ = gauge;

    int volume = qcu::config::lattice_volume_local();
    int complex_vector_length = Nd * volume * n_colors_ * n_colors_;
    
    assert(floatPrecision == kPrecisionDouble || floatPrecision == kPrecisionSingle ||
        floatPrecision == kPrecisionHalf);
    copyComplexVector_interface(fp64_gauge_, QcuPrecision::kPrecisionDouble, gauge_external_, floatPrecision, complex_vector_length);
    copyComplexVector_interface(fp32_gauge_, QcuPrecision::kPrecisionSingle, gauge_external_, floatPrecision, complex_vector_length);
    copyComplexVector_interface(fp16_gauge_, QcuPrecision::kPrecisionHalf, gauge_external_, floatPrecision, complex_vector_length);
}

void Qcu::push_back_fermion(void* fermionOut, void* fermionIn) {
    if (fermion_in_vec_.size() >= m_input_ || fermion_out_vec_.size() >= m_input_) {
        errorQcu("Fermion queue is full\n");
    }
    fermion_in_vec_.push_back(fermionIn);
    fermion_out_vec_.push_back(fermionOut);
}

// void Qcu::solve_fermions(int max_iteration, double max_precision) {
//     if (underlying_args_.out_float_precision == QcuPrecision::kPrecisionDouble) {
//         solve_fermions_template_function<double>(max_iteration, max_precision);
//     } else if (underlying_args_.out_float_precision == QcuPrecision::kPrecisionSingle) {
//         solve_fermions_template_function<float>(max_iteration, max_precision);
//     } else if (underlying_args_.out_float_precision == QcuPrecision::kPrecisionHalf) {
//         solve_fermions_template_function<half>(max_iteration, max_precision);
//     } else {
//         errorQcu("Unsupported float precision\n");
//     }
// }

// template <typename OutputFloat>
// void Qcu::solve_fermions_template_function(int max_iteration, double max_precision) {
//
void Qcu::solve_fermions(int max_iteration, double max_precision) {
    const int vol = qcu::config::lattice_volume_local();
    const int colorSpinor_len = n_spin_ * n_colors_;

    size_t io_float_size;
    size_t compute_float_size;
    if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionHalf) {
        compute_float_size = sizeof(half);
    } else if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionSingle) {
        compute_float_size = sizeof(float);
    } else if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionDouble) {
        compute_float_size = sizeof(double);
    } else {
            errorQcu("Unsupported float precision\n");
    }
    if (underlying_args_.out_float_precision == QcuPrecision::kPrecisionHalf) {
        io_float_size = sizeof(half);
    } else if (underlying_args_.out_float_precision == QcuPrecision::kPrecisionSingle) {
        io_float_size = sizeof(float);
    } else if (underlying_args_.out_float_precision == QcuPrecision::kPrecisionDouble) {
        io_float_size = sizeof(double);
    } else {
            errorQcu("Unsupported float precision\n");
    }

    if (m_input_ != fermion_in_vec_.size()) {
        errorQcu("number of fermion is different from mInput\n");
    } else {
        printf("numbers matched, now begin bicg\n");
    }
    std::vector<void*> fermionIn_queue_odd(fermion_in_vec_.size());
    std::vector<void*> fermionOut_queue_odd(fermion_out_vec_.size());
    for (int i = 0; i < fermion_in_vec_.size(); i++) {
        fermionIn_queue_odd[i] = static_cast<char*>(fermion_in_vec_[i]) + colorSpinor_len * vol / 2 * io_float_size * 2;
        fermionOut_queue_odd[i] = static_cast<char*>(fermion_out_vec_[i]) + colorSpinor_len * vol / 2 * io_float_size * 2;
    }

    void* fermionIn_MRHS_even = fermion_in_mrhs_;
    void* fermionIn_MRHS_odd = static_cast<char*>(fermion_in_mrhs_) + colorSpinor_len * m_input_ * vol / 2 * compute_float_size * 2;

    QcuPrecision compute_precision = underlying_args_.compute_float_precision;
    QcuPrecision reduction_precision = compute_precision;
    if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionHalf) {
        reduction_precision = QcuPrecision::kPrecisionSingle;
    }
    // gather even
    CHECK_CUDA(cudaMemcpy(d_lookup_table_in_,  fermion_in_vec_.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice));
    colorSpinorGather(
        fermionIn_MRHS_even,
        compute_precision,
        d_lookup_table_in_,
        underlying_args_.out_float_precision,
        *qcu::config::get_lattice_desc_ptr(), n_colors_, m_input_, NULL, n_spin_);

    // gather odd
    CHECK_CUDA(cudaMemcpy(d_lookup_table_in_, fermionIn_queue_odd.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice));
    colorSpinorGather(
        fermionIn_MRHS_odd, underlying_args_.compute_float_precision,
        d_lookup_table_in_, underlying_args_.out_float_precision,
        *qcu::config::get_lattice_desc_ptr(), n_colors_, m_input_, NULL, n_spin_);

    // SOLVE
    void* gauge;
    if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionDouble) {
        gauge = fp64_gauge_;
    }
    else if (underlying_args_.compute_float_precision == QcuPrecision::kPrecisionSingle) {
        gauge = fp32_gauge_;
    }
    else {
        gauge = fp16_gauge_;
    }

    std::shared_ptr<qcu::FermionGhost<Nd>> fermion_ghost_ptr
        = std::make_shared<qcu::FermionGhost<Nd>>(
            underlying_args_.lattice_desc_ptr,
            config::get_mpi_separated_mask(),
            n_colors_,
            m_input_,
            underlying_args_.compute_float_precision);

    qcu::solver::BiCGStabParam param{
        .nColor         = n_colors_,
        .mInput         = m_input_,
        .Nspin          = 4,
        .t_boudary      = t_boundary_,
        .staggered_phase= staggered_phase_,
        .kappa          = kappa_,
        .output_x_mrhs  = fermion_out_mrhs_,
        .input_b_mrhs   = fermion_in_mrhs_,
        .gauge          = gauge,
        .lattDesc       = &(underlying_args_.lattice_desc_ptr),
        .procDesc       = &(underlying_args_.process_desc_ptr),
        .streams = config::get_qcu_streams(),
        .fermion_ghost_ = fermion_ghost_ptr,
        .use_combined_residual = residual_combine_flag_,
        .use_tensor_core = tensor_core_flag_
    };

    solver::ApplyBicgStab(param, reduction_precision, compute_precision, max_iteration, max_precision);
    // scatter
    void* fermionOut_MRHS_even = fermion_out_mrhs_;
    void* fermionOut_MRHS_odd = static_cast<char*>(fermion_out_mrhs_) + colorSpinor_len * m_input_ * vol / 2 * compute_float_size * 2;
    // scatter even
    CHECK_CUDA(cudaMemcpy(d_lookup_table_out_, fermion_out_vec_.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice));

    colorSpinorScatter( d_lookup_table_out_,   underlying_args_.out_float_precision,
                        fermionOut_MRHS_even, underlying_args_.compute_float_precision,
                        *qcu::config::get_lattice_desc_ptr(), n_colors_, m_input_, NULL, n_spin_);
    // scatter odd
    CHECK_CUDA(cudaMemcpy(d_lookup_table_out_, fermionOut_queue_odd.data(), sizeof(void*) * m_input_, cudaMemcpyHostToDevice));
        colorSpinorScatter( d_lookup_table_out_,  underlying_args_.out_float_precision,
                            fermionOut_MRHS_odd, underlying_args_.compute_float_precision,
                            *qcu::config::get_lattice_desc_ptr(), n_colors_, m_input_, NULL, n_spin_);
    CHECK_CUDA(cudaStreamSynchronize(NULL));
    fermion_in_vec_.clear();
    fermion_out_vec_.clear();
}

// template <typename Float_>
template <typename Float_>
void Qcu::read_gauge_from_file (const char* file_path, void* data_ptr) {
    int mpi_rank;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    std::vector<int> global_latt_desc_vec = config::get_latt_desc();
    std::vector<int> mpi_desc_vec = config::get_mpi_desc();
    assert(global_latt_desc_vec.size() == mpi_desc_vec.size());
    std::vector<int> local_latt_desc_vec{
        global_latt_desc_vec[X_DIM] / mpi_desc_vec[X_DIM],
        global_latt_desc_vec[Y_DIM] / mpi_desc_vec[Y_DIM],
        global_latt_desc_vec[Z_DIM] / mpi_desc_vec[Z_DIM],
        global_latt_desc_vec[T_DIM] / mpi_desc_vec[T_DIM]
    };
    qcu::io::GaugeStorage<std::complex<Float_>> gauge(global_latt_desc_vec, n_colors_);
    qcu::io::GaugeReader<Float_> reader(mpi_rank, mpi_desc_vec);
    reader.read(file_path, gauge);

    size_t gauge_length = config::lattice_volume_local() * Nd * n_colors_ * n_colors_;

    Complex<Float_>* d_unpreconditioned = nullptr;
    CHECK_CUDA(cudaMalloc(&d_unpreconditioned, sizeof(Complex<Float_>) * gauge_length));
    CHECK_CUDA(cudaMemcpy(d_unpreconditioned, gauge.data_ptr(), sizeof(Complex<Float_>) * gauge_length, cudaMemcpyHostToDevice));
    qcu::GaugeEOPreconditioner<Float_> preconditioner;
    preconditioner.reverse(static_cast<Complex<Float_>*>(data_ptr),
                            d_unpreconditioned,
                            local_latt_desc_vec,
                            n_colors_ * n_colors_,
                            4,
                            nullptr);
    CHECK_CUDA(cudaFree(d_unpreconditioned));
}
void Qcu::set_staggered_phase (QcuStaggeredPhase staggered_phase) {
    staggered_phase_ = staggered_phase;
}

// 启动scatter
void Qcu::begin_scatter() {
    cudaStream_t stream = config::get_qcu_streams()[8];

    colorSpinorScatter(d_lookup_table_out_, underlying_args_.out_float_precision, fermion_out_mrhs_,
            underlying_args_.compute_float_precision, *config::get_lattice_desc_ptr(), n_colors_, m_input_, stream, n_spin_);
    CHECK_CUDA(cudaDeviceSynchronize());

    fermion_in_vec_.clear();
    fermion_out_vec_.clear();
}
// 启动gather
void Qcu::begin_gather() {
    if (fermion_in_vec_.size() != m_input_ || fermion_out_vec_.size() != m_input_) {
        errorQcu("Fermion queue is not full\n");
    }
    cudaStream_t stream = config::get_qcu_streams()[8];
    CHECK_CUDA(cudaMemcpy(d_lookup_table_in_, fermion_in_vec_.data(), sizeof(void*) * fermion_in_vec_.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lookup_table_out_, fermion_out_vec_.data(), sizeof(void*) * fermion_in_vec_.size(), cudaMemcpyHostToDevice));

    colorSpinorGather(fermion_in_mrhs_, underlying_args_.compute_float_precision,
        d_lookup_table_in_, underlying_args_.out_float_precision, *qcu::config::get_lattice_desc_ptr(),
        n_colors_, m_input_, stream, n_spin_);
    CHECK_CUDA(cudaStreamSynchronize(stream));
}


template void Qcu::read_gauge_from_file<double> (const char* file_path, void* data_ptr);
template void Qcu::read_gauge_from_file<float> (const char* file_path, void* data_ptr);
// template void Qcu::read_gauge_from_file<half> (const char* file_path, void* data_ptr);
}  // namespace qcu
