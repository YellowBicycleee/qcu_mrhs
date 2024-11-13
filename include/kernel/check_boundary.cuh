//
// Created by wangj on 2024/11/13.
//

#pragma once

#include "point/qcu_point.cuh"

namespace qcu::kernel {
enum class QcuBoundary {
    kQcuSmallBoundary,
    kQcuLargeBoundary,
    kQcuCenter
};

QCU_DEVICE
QcuBoundary check_boundary_with_eo_precondition (QcuLattDesc& latt_desc, Point& point, int dim, int multiprocess) {

    if (dim < 0 || dim > 1) {
        printf("Fatal, in function %s, error dim %d, must be one of (x-0, y-1, z-2, t-3)\n", __func__, dim);
        exit(-1);
    }

    if (multiprocess & (1 << dim)) {
        return QcuBoundary::kQcuCenter; // no multiprocess on this dimension
    }

    switch (dim) {
        case X_DIM:
            {
                if (latt_desc.X() == 0
                    && point.Parity() == ((point.Y() + point.Z() + point.T()) & 1))
                {
                    return QcuBoundary::kQcuSmallBoundary;
                }
                else if (latt_desc.X() == point.X() / 2 - 1
                    && point.Parity() != ((point.Y() + point.Z() + point.T()) & 1))
                {
                    return QcuBoundary::kQcuLargeBoundary;
                }
                else return QcuBoundary::kQcuCenter;
            }
            break;
        case Y_DIM:
            {
                if (latt_desc.Y() == 0) { return QcuBoundary::kQcuSmallBoundary; }
                else if (latt_desc.Y() == point.Y() - 1) { return QcuBoundary::kQcuLargeBoundary; }
                else return QcuBoundary::kQcuCenter;
            }
            break;
        case Z_DIM:
            {
                if (latt_desc.Z() == 0) { return QcuBoundary::kQcuSmallBoundary; }
                else if (latt_desc.Z() == point.Z() - 1) { return QcuBoundary::kQcuLargeBoundary; }
                else return QcuBoundary::kQcuCenter;
            }
            break;
        case T_DIM:
            {
                if (latt_desc.T() == 0) { return QcuBoundary::kQcuSmallBoundary; }
                else if (latt_desc.T() == point.T() - 1) { return QcuBoundary::kQcuLargeBoundary; }
                else return QcuBoundary::kQcuCenter;
            }
            break;
        default:
            printf("Fatal, in function %s, error dim %d, must be one of (x-0, y-1, z-2, t-3)\n", __func__, dim);
            exit(-1);
    }
    return QcuBoundary::kQcuCenter;
}

}