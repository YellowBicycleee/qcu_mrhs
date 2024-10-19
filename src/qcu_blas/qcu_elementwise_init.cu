#include "kernel/element_wise/elementwise_init.cuh"
#include "qcu_blas_public.h"
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_elementwise_init.h"

namespace qcu::qcu_blas {

template <typename _Tp>
using ElementwiseInitArg = typename qcu::qcu_blas::ElementwiseInit<_Tp>::ElementwiseInitArgument;

template <typename _Tp>
void ElementwiseInit<_Tp>::operator()(ElementwiseInitArgument arg) 
{
  int threads_per_block = std::min(512, maxThreadsPerBlock);
  int blocks_per_grid   = std::min(div_ceil(arg.vec_len, threads_per_block),
                                              maxGridSize);

    qcu::device::kernel::elementwise_init_kernel<_Tp> 
        <<<blocks_per_grid, threads_per_block, 0, arg.stream>>>
                        (arg.res, arg.val, arg.vec_len);  

}

template struct ElementwiseInit<Complex<half>>;
template struct ElementwiseInit<Complex<float>>;
template struct ElementwiseInit<Complex<double>>;
}