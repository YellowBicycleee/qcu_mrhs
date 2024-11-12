#pragma once

#include "qcu_public.h"
#include "qcu.h"
#include "qcu_helper.h"

namespace qcu {

// in every process, lattice size desc
struct QcuLattDesc {
    int data[MAX_DIM];
    int volume;

    QCU_HOST_DEVICE QcuLattDesc() : data {0, 0, 0, 0}, volume(0) {}

    QCU_HOST_DEVICE QcuLattDesc(int x, int y = 1, int z = 1, int t = 1)
        : data {x, y, z, t}, volume (x * y * z * t)
    {}

    QCU_HOST_DEVICE QcuLattDesc(QcuParam *param)
        : data {param->lattice_size[X_DIM], param->lattice_size[Y_DIM],
                param->lattice_size[Z_DIM], param->lattice_size[T_DIM]}
        , volume (param->lattice_size[X_DIM] * param->lattice_size[Y_DIM]
                * param->lattice_size[Z_DIM] * param->lattice_size[T_DIM]
        )  
    {}
    
    QCU_HOST_DEVICE int X() const { return data[X_DIM]; }
    QCU_HOST_DEVICE int Y() const { return data[Y_DIM]; }
    QCU_HOST_DEVICE int Z() const { return data[Z_DIM]; }
    QCU_HOST_DEVICE int T() const { return data[T_DIM]; }

    QCU_HOST_DEVICE int lattice_volume() const { return volume; }
    QCU_HOST_DEVICE int half_lattice_volume() const { return (volume >> 1); }
};

struct QcuProcDesc {  // process description, how many process in each dimension
    int data[MAX_DIM];
    int volume;
    QcuProcDesc(int x = 1, int y = 1, int z = 1, int t = 1)
        : data {x, y, z, t}, volume(x * y * z * t)
    {}

    QcuProcDesc(QcuGrid *grid)
        : data {grid->grid_size[X_DIM], grid->grid_size[Y_DIM]
            , grid->grid_size[Z_DIM], grid->grid_size[T_DIM] }
        , volume(grid->grid_size[X_DIM] * grid->grid_size[Y_DIM]
            * grid->grid_size[Z_DIM] * grid->grid_size[T_DIM])
    {}

    QCU_HOST_DEVICE int X() const { return data[X_DIM]; }
    QCU_HOST_DEVICE int Y() const { return data[Y_DIM]; }
    QCU_HOST_DEVICE int Z() const { return data[Z_DIM]; }
    QCU_HOST_DEVICE int T() const { return data[T_DIM]; }

    // x, y, z, t direction is separated
    QCU_HOST_DEVICE bool X_separated() const { return data[X_DIM] > 1; }
    QCU_HOST_DEVICE bool Y_separated() const { return data[Y_DIM] > 1; }
    QCU_HOST_DEVICE bool Z_separated() const { return data[Z_DIM] > 1; }
    QCU_HOST_DEVICE bool T_separated() const { return data[T_DIM] > 1; }

    // volume
    QCU_HOST_DEVICE int process_volume() const { return volume; }
};
}  // namespace qcu
