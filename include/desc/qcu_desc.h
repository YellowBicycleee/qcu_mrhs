#pragma once

#include <cassert>
#include <cstdio>

#include "qcu_public.h"
#include "qcu.h"
// clang-format off

namespace qcu {
// in every process, lattice size desc
struct QcuLattDesc {
    int dims[MAX_DIM];

    QcuLattDesc(int x, int y = 1, int z = 1, int t = 1) {
        dims[X_DIM] = x;
        dims[Y_DIM] = y;
        dims[Z_DIM] = z;
        dims[T_DIM] = t;
    }

    QcuLattDesc(QcuParam *param) {
        dims[X_DIM] = param->lattice_size[X_DIM];
        dims[Y_DIM] = param->lattice_size[Y_DIM];
        dims[Z_DIM] = param->lattice_size[Z_DIM];
        dims[T_DIM] = param->lattice_size[T_DIM];
    }
    int latticeDimLength(int dim) const { return dims[dim]; }
    int latticeVolumn() const { return dims[X_DIM] * dims[Y_DIM] * dims[Z_DIM] * dims[T_DIM]; }
    int X() const { return dims[X_DIM]; }
    int Y() const { return dims[Y_DIM]; }
    int Z() const { return dims[Z_DIM]; }
    int T() const { return dims[T_DIM]; }
};

struct QcuProcDesc {  // process description
    int dims[MAX_DIM];
    QcuProcDesc(int x = 1, int y = 1, int z = 1, int t = 1) {
        dims[X_DIM] = x;
        dims[Y_DIM] = y;
        dims[Z_DIM] = z;
        dims[T_DIM] = t;
    }

    QcuProcDesc(QcuGrid *grid) {
        dims[X_DIM] = grid->grid_size[X_DIM];
        dims[Y_DIM] = grid->grid_size[Y_DIM];
        dims[Z_DIM] = grid->grid_size[Z_DIM];
        dims[T_DIM] = grid->grid_size[T_DIM];
    }

    int dimProcess(int dim) const { return dims[dim]; }
    int X() const { return dims[X_DIM]; }
    int Y() const { return dims[Y_DIM]; }
    int Z() const { return dims[Z_DIM]; }
    int T() const { return dims[T_DIM]; }
};
}  // namespace qcu
