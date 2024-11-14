#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "desc/qcu_desc.h"
#include "precondition/even_odd_precondition.h"
#include "qcu.h"
#include "qcu_interface.h"
#include "qcu_public.h"
#include "qcu_config/qcu_config.h"

static qcu::Qcu *qcu_ptr = nullptr;

static void check_qcu_ptr() {
    if (qcu_ptr == nullptr) {
        errorQcu("Qcu is not initialized\n");
    }
}

void initGridSize(QcuGrid *grid, QcuParam *param, int n_color, int m_rhs, int inputFloatPrecision,
                  int dslashFloatPrecision) {

    int Lx = param->lattice_size[X_DIM];
    int Ly = param->lattice_size[Y_DIM];
    int Lz = param->lattice_size[Z_DIM];
    int Lt = param->lattice_size[T_DIM];
    int Gx = grid->grid_size[X_DIM];
    int Gy = grid->grid_size[Y_DIM];
    int Gz = grid->grid_size[Z_DIM];
    int Gt = grid->grid_size[T_DIM];

    qcu::config::set_config(Lx, Ly, Lz, Lt, Gx, Gy, Gz, Gt);
    qcu_ptr = new qcu::Qcu(Lx, Ly, Lz, Lt, Gx, Gy, Gz, Gt, (QcuPrecision)inputFloatPrecision,
                         (QcuPrecision)dslashFloatPrecision, n_color, m_rhs);
    qcu::config::init_streams();
}

void pushBackFermions(void *fermionOut, void *fermionIn) {
    check_qcu_ptr();
    qcu_ptr->push_back_fermion(fermionOut, fermionIn);
}

void loadQcuGauge(void *gauge, int floatPrecision) { 
    check_qcu_ptr();
    qcu_ptr->load_gauge(gauge, (QcuPrecision)floatPrecision);
}

void getDslash(int dslashType, double mass) { 
    check_qcu_ptr();
    qcu_ptr->get_dslash((DslashType)dslashType, mass);
}

void start_dslash(int parity, int daggerFlag) {
    check_qcu_ptr();
    qcu_ptr->start_dslash(parity, (bool)daggerFlag);
}
void mat_Qcu(int daggerFlag) {
    check_qcu_ptr();
    qcu_ptr->mat_qcu((bool)daggerFlag);
}

void finalizeQcu() {
    // check_qcu_ptr();
    delete qcu_ptr;
    qcu_ptr = nullptr;
    qcu::config::destroy_streams();
}

void qcuInvert(int max_iteration, double max_precison) {
    check_qcu_ptr();
    qcu_ptr->solve_fermions(max_iteration, max_precison);
}

// 奇偶预处理接口
void gauge_eo_precondition(void *prec_gauge, void *non_prec_gauge, int precision) {
    check_qcu_ptr();
    assert(prec_gauge != non_prec_gauge);

    Latt_Desc total_latt_desc;
    Latt_Desc local_latt_desc;
    MPI_Desc mpi_desc;

    const qcu::QcuLattDesc* qcu_lattice_desc = qcu::config::get_lattice_desc_ptr();
    const qcu::QcuProcDesc* qcu_process_desc = qcu::config::get_process_desc_ptr();

#pragma unroll
    for (int i = X_DIM; i < Nd; ++i) {
        total_latt_desc.data[i] = qcu_lattice_desc->data[i];
        mpi_desc.data[i] = qcu_process_desc->data[i];

        local_latt_desc.data[i] = total_latt_desc.data[i] / mpi_desc.data[i];
    }

    int site_vec_len = qcu_ptr->color() * qcu_ptr->color();

    if (precision == QcuPrecision::kPrecisionDouble) {
        qcu::GaugeEOPreconditioner<double> preconditioner;
        preconditioner.apply(static_cast<Complex<double> *>(prec_gauge), static_cast<Complex<double> *>(non_prec_gauge),
            local_latt_desc, site_vec_len);
    } else if (precision == QcuPrecision::kPrecisionSingle) {
        qcu::GaugeEOPreconditioner<float> preconditioner;
        preconditioner.apply(static_cast<Complex<float> *>(prec_gauge), static_cast<Complex<float> *>(non_prec_gauge),
            local_latt_desc, site_vec_len);
    } else if (precision == QcuPrecision::kPrecisionHalf) {
        qcu::GaugeEOPreconditioner<half> preconditioner;
        preconditioner.apply(static_cast<Complex<half> *>(prec_gauge), static_cast<Complex<half> *>(non_prec_gauge),
            local_latt_desc, site_vec_len);
    } else {
        errorQcu("UNDEFINED precision");
    }
}
void gauge_reverse_eo_precondition(void *non_prec_gauge, void *prec_gauge, int precision) {
    check_qcu_ptr();
    assert(prec_gauge != non_prec_gauge);

    Latt_Desc total_latt_desc;
    Latt_Desc local_latt_desc;
    MPI_Desc mpi_desc;

    const qcu::QcuLattDesc* qcu_lattice_desc = qcu::config::get_lattice_desc_ptr();
    const qcu::QcuProcDesc* qcu_process_desc = qcu::config::get_process_desc_ptr();

#pragma unroll
    for (int i = X_DIM; i < Nd; ++i) {
        total_latt_desc.data[i] = qcu_lattice_desc->data[i];
        mpi_desc.data[i] = qcu_process_desc->data[i];

        local_latt_desc.data[i] = total_latt_desc.data[i] / mpi_desc.data[i];
    }

    int site_vec_len = qcu_ptr->color() * qcu_ptr->color();

    if (precision == QcuPrecision::kPrecisionDouble) {
        qcu::GaugeEOPreconditioner<double> preconditioner;
        preconditioner.reverse(static_cast<Complex<double> *>(non_prec_gauge), static_cast<Complex<double> *>(prec_gauge),
                           local_latt_desc, site_vec_len);
    } else if (precision == QcuPrecision::kPrecisionSingle) {
    qcu::GaugeEOPreconditioner<float> preconditioner;
    preconditioner.reverse(static_cast<Complex<float> *>(non_prec_gauge), static_cast<Complex<float> *>(prec_gauge),
                           local_latt_desc, site_vec_len);
  } else if (precision == QcuPrecision::kPrecisionHalf) {
    qcu::GaugeEOPreconditioner<half> preconditioner;
    preconditioner.reverse(static_cast<Complex<half> *>(non_prec_gauge), static_cast<Complex<half> *>(prec_gauge),
                           local_latt_desc, site_vec_len);
  } else {
    errorQcu("UNDEFINED precision");
  }
}

void read_gauge_from_file (void* gauge, const char* file_path_prefix) {
    check_qcu_ptr();
    qcu_ptr->read_gauge_from_file(file_path_prefix, gauge);
}