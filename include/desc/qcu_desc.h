#pragma once

#include <cassert>
#include <cstdio>

#include "qcu_enum.h"
// clang-format off

namespace qcu {
// in every process, lattice size desc
struct QcuLattDesc {
    int dim[MAX_DIM];

    QcuLattDesc(int x, int y = 1, int z = 1, int t = 1) {
        dim[X_DIM] = x;
        dim[Y_DIM] = y;
        dim[Z_DIM] = z;
        dim[T_DIM] = t;
    }

    QcuLattDesc(QcuParam *param) {
        dim[X_DIM] = param->lattice_size[X_DIM];
        dim[Y_DIM] = param->lattice_size[Y_DIM];
        dim[Z_DIM] = param->lattice_size[Z_DIM];
        dim[T_DIM] = param->lattice_size[T_DIM];
    }
    int latticeDimLength(int dim) { return dim[dim]; }
    int X() { return dim[X_DIM]; }
    int Y() { return dim[Y_DIM]; }
    int Z() { return dim[Z_DIM]; }
    int T() { return dim[T_DIM]; }
};

struct QcuProcDesc {  // process description
    int dim[MAX_DIM];
    QcuProcDesc(int x = 1, int y = 1, int z = 1, int t = 1) {
        dim[X_DIM] = x;
        dim[Y_DIM] = y;
        dim[Z_DIM] = z;
        dim[T_DIM] = t;
    }

    QcuProcDesc(QcuGrid *grid) {
        dim[X_DIM] = grid->grid_size[X_DIM];
        dim[Y_DIM] = grid->grid_size[Y_DIM];
        dim[Z_DIM] = grid->grid_size[Z_DIM];
        dim[T_DIM] = grid->grid_size[T_DIM];
    }

    int dimProcess(int dim) { return dim[dim]; }
    int X() { return dim[X_DIM]; }
    int Y() { return dim[Y_DIM]; }
    int Z() { return dim[Z_DIM]; }
    int T() { return dim[T_DIM]; }
};
}  // namespace qcu
