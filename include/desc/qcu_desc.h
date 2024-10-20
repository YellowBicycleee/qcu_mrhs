#pragma once

#include <cassert>
#include <cstdio>

#include "qcu_public.h"
#include "qcu.h"
// clang-format off

namespace qcu {
// in every process, lattice size desc
struct QcuLattDesc {
    int data[MAX_DIM];

    QcuLattDesc(int x, int y = 1, int z = 1, int t = 1) {
        data[X_DIM] = x;
        data[Y_DIM] = y;
        data[Z_DIM] = z;
        data[T_DIM] = t;
    }

    QcuLattDesc(QcuParam *param) {
        data[X_DIM] = param->lattice_size[X_DIM];
        data[Y_DIM] = param->lattice_size[Y_DIM];
        data[Z_DIM] = param->lattice_size[Z_DIM];
        data[T_DIM] = param->lattice_size[T_DIM];
    }
    int latticeDimLength(int dim) const { return data[dim]; }
    int latticeVolumn() const { return data[X_DIM] * data[Y_DIM] * data[Z_DIM] * data[T_DIM]; }
    int X() const { return data[X_DIM]; }
    int Y() const { return data[Y_DIM]; }
    int Z() const { return data[Z_DIM]; }
    int T() const { return data[T_DIM]; }
};

struct QcuProcDesc {  // process description
    int data[MAX_DIM];
    QcuProcDesc(int x = 1, int y = 1, int z = 1, int t = 1) {
        data[X_DIM] = x;
        data[Y_DIM] = y;
        data[Z_DIM] = z;
        data[T_DIM] = t;
    }

    QcuProcDesc(QcuGrid *grid) {
        data[X_DIM] = grid->grid_size[X_DIM];
        data[Y_DIM] = grid->grid_size[Y_DIM];
        data[Z_DIM] = grid->grid_size[Z_DIM];
        data[T_DIM] = grid->grid_size[T_DIM];
    }

    int dimProcess(int dim) const { return data[dim]; }
    int X() const { return data[X_DIM]; }
    int Y() const { return data[Y_DIM]; }
    int Z() const { return data[Z_DIM]; }
    int T() const { return data[T_DIM]; }
};
}  // namespace qcu
