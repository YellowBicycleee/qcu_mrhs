// use this file to define the matrix loader for the threadblock

#pragma once
#include "complex/qcu_complex.h"
#include "qcu_helper.h"

namespace qcu {
namespace matrix {

template <typename _FloatType>
using Complex = ::Complex<_FloatType>;

enum class QcuMatrixMajor {
    kQcuRowMajor,
    kQcuColMajor
};

// 是否有必要写dataloader from global memory to reg
template <
    typename _FloatType = double,
    QcuMatrixMajor _MatrixMajor = QcuMatrixMajor::kQcuRowMajor,
    int _BLK_M = 16,
    int _BLK_N = 16
>
class ComplexLgs2Reg;

template <>
class ComplexLgs2Reg<double> {
public:
    QCU_DEVICE void operator() (Complex<double>* glb_mem, int m, int n,
        Complex<double>* reg_mem, int glb_start_m, int glb_start_n)
    {}
};

}
}