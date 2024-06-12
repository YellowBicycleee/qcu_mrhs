#pragma once

#include <cassert>
#include <cstdio>

#include "qcu_enum.h"
#include "qcu_macro.h"
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
    int latticeDimLength(int dim) { return dims[dim]; }
    int X() { return dims[X_DIM]; }
    int Y() { return dims[Y_DIM]; }
    int Z() { return dims[Z_DIM]; }
    int T() { return dims[T_DIM]; }
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

    int dimProcess(int dim) { return dims[dim]; }
    int X() { return dims[X_DIM]; }
    int Y() { return dims[Y_DIM]; }
    int Z() { return dims[Z_DIM]; }
    int T() { return dims[T_DIM]; }
};
}  // namespace qcu
