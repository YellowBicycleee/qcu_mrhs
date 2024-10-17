#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "desc/qcu_desc.h"
#include "precondition/even_odd_precondition.h"
#include "qcu.h"
#include "qcu_interface.h"
#include "qcu_public.h"

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
  qcu_ptr = new qcu::Qcu(Lx, Ly, Lz, Lt, Gx, Gy, Gz, Gt, (QCU_PRECISION)inputFloatPrecision,
                         (QCU_PRECISION)dslashFloatPrecision, n_color, m_rhs);
}

void pushBackFermions(void *fermionOut, void *fermionIn) {
  check_qcu_ptr();
  qcu_ptr->pushBackFermions(fermionOut, fermionIn);
}

void loadQcuGauge(void *gauge, int floatPrecision) { 
  check_qcu_ptr();
  qcu_ptr->loadGauge(gauge, (QCU_PRECISION)floatPrecision); 
}

void getDslash(int dslashType, double mass) { 
  check_qcu_ptr();
  qcu_ptr->getDslash((DSLASH_TYPE)dslashType, mass); 
}

void start_dslash(int parity, int daggerFlag) {
  check_qcu_ptr();
  qcu_ptr->startDslash(parity, (bool)daggerFlag);
}
void mat_Qcu(int daggerFlag) {
  check_qcu_ptr();
  qcu_ptr->MatQcu((bool)daggerFlag);
}

void finalizeQcu() {
  // check_qcu_ptr();
  delete qcu_ptr;
  qcu_ptr = nullptr;
}

void qcuInvert(int max_iteration, double max_precison) {
  check_qcu_ptr();
  qcu_ptr->solveFermions(max_iteration, max_precison);
}

// 奇偶预处理接口
void gauge_eo_precondition(void *prec_gauge, void *non_prec_gauge, int precision) {
  check_qcu_ptr();
  assert(prec_gauge != non_prec_gauge);

  Latt_Desc total_latt_desc;
  Latt_Desc local_latt_desc;
  MPI_Desc mpi_desc;

  qcu::QcuLattDesc qcuLattDesc = qcu_ptr->lattDesc();
  qcu::QcuProcDesc qcuProcDesc = qcu_ptr->procDesc();

#pragma unroll
  for (int i = X_DIM; i < Nd; ++i) {
    total_latt_desc.data[i] = qcuLattDesc.dims[i];
    mpi_desc.data[i] = qcuProcDesc.dims[i];

    local_latt_desc.data[i] = total_latt_desc.data[i] / mpi_desc.data[i];
  }

  int site_vec_len = qcu_ptr->color() * qcu_ptr->color();

  if (precision == QCU_PRECISION::QCU_DOUBLE_PRECISION) {
    qcu::GaugeEOPreconditioner<double> preconditioner;
    preconditioner.apply(static_cast<Complex<double> *>(prec_gauge), static_cast<Complex<double> *>(non_prec_gauge),
                         local_latt_desc, site_vec_len);
  } else if (precision == QCU_PRECISION::QCU_SINGLE_PRECISION) {
    qcu::GaugeEOPreconditioner<float> preconditioner;
    preconditioner.apply(static_cast<Complex<float> *>(prec_gauge), static_cast<Complex<float> *>(non_prec_gauge),
                         local_latt_desc, site_vec_len);
  } else if (precision == QCU_PRECISION::QCU_HALF_PRECISION) {
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

  qcu::QcuLattDesc qcuLattDesc = qcu_ptr->lattDesc();
  qcu::QcuProcDesc qcuProcDesc = qcu_ptr->procDesc();

#pragma unroll
  for (int i = X_DIM; i < Nd; ++i) {
    total_latt_desc.data[i] = qcuLattDesc.dims[i];
    mpi_desc.data[i] = qcuProcDesc.dims[i];

    local_latt_desc.data[i] = total_latt_desc.data[i] / mpi_desc.data[i];
  }

  int site_vec_len = qcu_ptr->color() * qcu_ptr->color();

  if (precision == QCU_PRECISION::QCU_DOUBLE_PRECISION) {
    qcu::GaugeEOPreconditioner<double> preconditioner;
    preconditioner.reverse(static_cast<Complex<double> *>(non_prec_gauge), static_cast<Complex<double> *>(prec_gauge),
                           local_latt_desc, site_vec_len);
  } else if (precision == QCU_PRECISION::QCU_SINGLE_PRECISION) {
    qcu::GaugeEOPreconditioner<float> preconditioner;
    preconditioner.reverse(static_cast<Complex<float> *>(non_prec_gauge), static_cast<Complex<float> *>(prec_gauge),
                           local_latt_desc, site_vec_len);
  } else if (precision == QCU_PRECISION::QCU_HALF_PRECISION) {
    qcu::GaugeEOPreconditioner<half> preconditioner;
    preconditioner.reverse(static_cast<Complex<half> *>(non_prec_gauge), static_cast<Complex<half> *>(prec_gauge),
                           local_latt_desc, site_vec_len);
  } else {
    errorQcu("UNDEFINED precision");
  }
}

void read_gauge_from_file (void* gauge, const char* file_path_prefix) {
  check_qcu_ptr();
  qcu_ptr->readGaugeFromFile(file_path_prefix, gauge);
}