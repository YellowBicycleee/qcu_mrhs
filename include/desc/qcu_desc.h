#pragma once

#include <cassert>
#include <cstdio>

#include "qcu_enum.h"

namespace qcu {
// in every process, lattice size desc
template <int _Ndim>
struct QcuLattDesc {
    int dim[_Ndim];
    QcuLattDesc(int x, int y = 1, int z = 1, int t = 1) {
        dim[X_DIM] = x;
        if (Y_DIM < _Ndim) dim[Y_DIM] = y;
        if (Z_DIM < _Ndim) dim[Z_DIM] = z;
        if (T_DIM < _Ndim) dim[T_DIM] = t;
    }
    QcuLattDesc(QcuParam *param) {
        dim[X_DIM] = param->lattice_size[X_DIM];
        if (Y_DIM < _Ndim) dim[Y_DIM] = param->lattice_size[Y_DIM];
        if (Z_DIM < _Ndim) dim[Z_DIM] = param->lattice_size[Z_DIM];
        if (T_DIM < _Ndim) dim[T_DIM] = param->lattice_size[T_DIM];
    }

    int X() { return dim[X_DIM]; }
    int Y() {
        if (_Ndim > Y_DIM)
            return dim[Y_DIM];
        else {
            printf("file %s line %d dim = %d\n", __FILE__, __LINE__, _Ndim);
            assert(0);
        }
    }
    int Z() {
        if (_Ndim > Z_DIM)
            return dim[Z_DIM];
        else {
            printf("file %s line %d dim = %d\n", __FILE__, __LINE__, _Ndim);
            assert(0);
        }
    }
    int T() {
        if (_Ndim > T_DIM)
            return dim[T_DIM];
        else {
            printf("file %s line %d dim = %d\n", __FILE__, __LINE__, _Ndim);
            assert(0);
        }
    }
};

template <int _Ndim>
struct QcuProcDesc {  // process description
    int dim[_Ndim];
    QcuProcDesc(int x = 1, int y = 1, int z = 1, int t = 1) {
        dim[X_DIM] = x;
        if (Y_DIM < _Ndim) dim[Y_DIM] = y;
        if (Z_DIM < _Ndim) dim[Z_DIM] = z;
        if (T_DIM < _Ndim) dim[T_DIM] = t;
    }
    QcuProcDesc(QcuGrid *grid) {
        dim[X_DIM] = grid->grid_size[X_DIM];
        if (Y_DIM < _Ndim) dim[Y_DIM] = grid->grid_size[Y_DIM];
        if (Z_DIM < _Ndim) dim[Z_DIM] = grid->grid_size[Z_DIM];
        if (T_DIM < _Ndim) dim[T_DIM] = grid->grid_size[T_DIM];
    }

    int X() { return dim[X_DIM]; }
    int Y() {
        if (_Ndim > Y_DIM)
            return dim[Y_DIM];
        else {
            printf("file %s line %d dim = %d\n", __FILE__, __LINE__, _Ndim);
            assert(0);
        }
    }

    int Z() {
        if (_Ndim > Z_DIM)
            return dim[Z_DIM];
        else {
            printf("file %s line %d dim = %d\n", __FILE__, __LINE__, _Ndim);
            assert(0);
        }
    }

    int T() {
        if (_Ndim > T_DIM)
            return dim[T_DIM];
        else {
            printf("file %s line %d dim = %d\n", __FILE__, __LINE__, _Ndim);
            assert(0);
        }
    }
};
}  // namespace qcu
