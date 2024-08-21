#include "kernel/element_wise/elementwise_div.cuh"
#include "qcu_blas_public.h"
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_elementwise_div.h"

namespace qcu::qcu_blas {

template <typename _Tp>
using ElementwiseDivArg = typename qcu::qcu_blas::ElementwiseDiv<_Tp>::ElementwiseDivArgument;

template <typename _Tp>
void ElementwiseDiv<_Tp>::operator()(ElementwiseDivArgument arg) 
{
  int threads_per_block = std::min(512, maxThreadsPerBlock);
  int blocks_per_grid   = std::min(div_ceil(arg.vec_len, threads_per_block),
                                              maxGridSize);

    qcu::device::kernel::elementwise_div_kernel<_Tp> 
        <<<blocks_per_grid, threads_per_block, 0, arg.stream>>>
                        (arg.res, arg.x, arg.y, arg.vec_len);  

}

template struct ElementwiseDiv<Complex<half>>;
template struct ElementwiseDiv<Complex<float>>;
template struct ElementwiseDiv<Complex<double>>;
}