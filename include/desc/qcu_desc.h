#pragma once

#include <cassert>
#include <cstdio>

#include "qcu_public.h"
#include "qcu.h"
#include "qcu_helper.h"
// clang-format off

namespace qcu {
// in every process, lattice size desc
struct QcuLattDesc {
    int data[MAX_DIM];
    const int volume;
    QCU_HOST_DEVICE
    QcuLattDesc(int x, int y = 1, int z = 1, int t = 1)
        : data {x, y, z, t}, volume (x * y * z * t) {}

    QCU_HOST_DEVICE
    QcuLattDesc(QcuParam *param)
        : data {param->lattice_size[X_DIM], param->lattice_size[Y_DIM],
                param->lattice_size[Z_DIM], param->lattice_size[T_DIM]}
        , volume (param->lattice_size[X_DIM] * param->lattice_size[Y_DIM]
                * param->lattice_size[Z_DIM] * param->lattice_size[T_DIM]
        )  
    {}
    QCU_HOST_DEVICE
    int latticeDimLength(int dim) const { return data[dim]; }

    QCU_HOST_DEVICE
    int latticeVolumn() const { return volume; }
    
    QCU_HOST_DEVICE
    int X() const { return data[X_DIM]; }

    QCU_HOST_DEVICE
    int Y() const { return data[Y_DIM]; }

    QCU_HOST_DEVICE
    int Z() const { return data[Z_DIM]; }

    QCU_HOST_DEVICE
    int T() const { return data[T_DIM]; }
};

struct QcuProcDesc {  // process description
    int data[MAX_DIM];
    QcuProcDesc(int x = 1, int y = 1, int z = 1, int t = 1) : data {x, y, z, t} {}

    QcuProcDesc(QcuGrid *grid)
        : data {grid->grid_size[X_DIM], grid->grid_size[Y_DIM]
            , grid->grid_size[Z_DIM], grid->grid_size[T_DIM] }
    {}

    int dimProcess(int dim) const { return data[dim]; }
    int X() const { return data[X_DIM]; }
    int Y() const { return data[Y_DIM]; }
    int Z() const { return data[Z_DIM]; }
    int T() const { return data[T_DIM]; }
};
}  // namespace qcu
