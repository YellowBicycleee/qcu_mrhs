//
// Created by wangj on 2024/8/26.
//

#include <cstdio>

#include "check_error/check_cuda.cuh"
#include "solver/bicgstab.cuh"

namespace qcu::solver {
template <QcuPrecision OutputPrecision>
static void InstantiateBicgStabIteratePrecision(BiCGStabParam& param, QcuPrecision iteratePrecision,
                                                int max_iteration = 1000, double max_precision = 1e-6)
{
    switch (iteratePrecision) {
        case QcuPrecision::kPrecisionDouble: {
            BiCGStabImpl<OutputPrecision, kPrecisionDouble> bicgstab(param, max_iteration, max_precision);
            bicgstab.solve();
        } break;
        case QcuPrecision::kPrecisionSingle: {
            BiCGStabImpl<OutputPrecision, QcuPrecision::kPrecisionSingle> bicgstab(param, max_iteration, max_precision);
            bicgstab.solve();
        } break;
        case QcuPrecision::kPrecisionHalf: {
            BiCGStabImpl<OutputPrecision, QcuPrecision::kPrecisionHalf> bicgstab(param, max_iteration, max_precision);
            bicgstab.solve();
        } break;
        default: {
            std::printf("error happened: wrong iteratePrecision\n");
            exit(1);
        } break;
    }
}

void ApplyBicgStab (BiCGStabParam& param,  QcuPrecision outputPrecision, QcuPrecision iteratePrecision,
                    int max_iteration, double max_precision)
{
    if (outputPrecision == QcuPrecision::kPrecisionDouble) {
        InstantiateBicgStabIteratePrecision<QcuPrecision::kPrecisionDouble>(param, iteratePrecision, max_iteration, max_precision);
    } else if (outputPrecision == QcuPrecision::kPrecisionSingle) {
        InstantiateBicgStabIteratePrecision<QcuPrecision::kPrecisionSingle>(param, iteratePrecision, max_iteration, max_precision);
    }
    else {
        std::printf("error happened: wrong outputPrecision, outputPrecision must be float or double\n");
        exit(1);
    }
}

}  // namespace qcu::solver