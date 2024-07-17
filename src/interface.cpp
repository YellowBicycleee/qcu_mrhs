#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "qcu.h"
#include "qcu_interface.h"
#include "qcu_macro.h"

static qcu::Qcu *qcu_ptr = nullptr;

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
    if (qcu_ptr) {
        qcu_ptr->pushBackFermions(fermionOut, fermionIn);
    } else {
        errorQcu("Qcu is not initialized\n");
    }
}



void loadQcuGauge(void *gauge, int floatPrecision) { qcu_ptr->loadGauge(gauge, (QCU_PRECISION)floatPrecision); }

void getDslash(int dslashType, double mass) {
    qcu_ptr->getDslash((DSLASH_TYPE)dslashType, mass);
}

void start_dslash(int parity, int daggerFlag) {
    if (qcu_ptr) {
        qcu_ptr->startDslash(parity, (bool)daggerFlag);
    } else {
        errorQcu("Qcu is not initialized\n");
    }
}

// void qcuInvert(void *x_vector, void *b_vector, void *gauge, QcuParam *param, double p_max_prec, double p_kappa) {
//     errorQcu("Not implemented yet\n");
// }

void finalizeQcu() {
    if (qcu_ptr != nullptr) {
        delete qcu_ptr;
    }
    qcu_ptr = nullptr;
}
