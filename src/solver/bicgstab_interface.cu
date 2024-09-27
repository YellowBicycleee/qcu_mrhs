//
// Created by wangj on 2024/8/26.
//

#include <cstdio>

#include "check_error/check_cuda.cuh"
#include "solver/bicgstab.cuh"

namespace qcu::solver {
template <QCU_PRECISION OutputPrecision>
static void InstantiateBicgStabIteratePrecision(BiCGStabParam& param, QCU_PRECISION iteratePrecision,
                                                int max_iteration = 1000, double max_precision = 1e-6)
{
  switch (iteratePrecision) {
    case QCU_DOUBLE_PRECISION: {
      BiCGStabImpl<OutputPrecision, QCU_DOUBLE_PRECISION> bicgstab(param, max_iteration, max_precision);
      bicgstab.solve();
    } break;
    case QCU_SINGLE_PRECISION: {
      BiCGStabImpl<OutputPrecision, QCU_SINGLE_PRECISION> bicgstab(param, max_iteration, max_precision);
      bicgstab.solve();
    } break;
    case QCU_HALF_PRECISION: {
      BiCGStabImpl<OutputPrecision, QCU_HALF_PRECISION> bicgstab(param, max_iteration, max_precision);
      bicgstab.solve();
    } break;
    default: {
      std::printf("error happened: wrong iteratePrecision\n");
      exit(1);
    } break;
  }
}

void ApplyBicgStab (BiCGStabParam& param,  QCU_PRECISION outputPrecision, QCU_PRECISION iteratePrecision,
                    int max_iteration, double max_precision)
{
  if (outputPrecision == QCU_DOUBLE_PRECISION) {
    InstantiateBicgStabIteratePrecision<QCU_DOUBLE_PRECISION>(param, iteratePrecision, max_iteration, max_precision);
  } else if (outputPrecision == QCU_SINGLE_PRECISION) {
    InstantiateBicgStabIteratePrecision<QCU_SINGLE_PRECISION>(param, iteratePrecision, max_iteration, max_precision);
  }
  else {
    std::printf("error happened: wrong outputPrecision, outputPrecision must be float or double\n");
    exit(1);
  }
}

}  // namespace qcu::solver