#include "kernel/element_wise/elementwise_mul.cuh"
#include "qcu_blas_public.h"
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_elementwise_mul.h"

namespace qcu::qcu_blas {

template <typename _Tp>
using ElementwiseMulArg = typename qcu::qcu_blas::ElementwiseMul<_Tp>::ElementwiseMulArgument;

template <typename _Tp>
void ElementwiseMul<_Tp>::operator()(ElementwiseMulArgument arg) 
{
  int threads_per_block = std::min(512, maxThreadsPerBlock);
  int blocks_per_grid   = std::min(div_ceil(arg.vec_len, threads_per_block),
                                              maxGridSize);

    qcu::device::kernel::elementwise_mul_kernel<_Tp> 
        <<<blocks_per_grid, threads_per_block, 0, arg.stream>>>
                        (arg.res, arg.x, arg.y, arg.vec_len);  

}

template struct ElementwiseMul<Complex<half>>;
template struct ElementwiseMul<Complex<float>>;
template struct ElementwiseMul<Complex<double>>;
}