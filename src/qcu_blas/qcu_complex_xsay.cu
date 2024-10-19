#include "kernel/element_wise/c_xsay.cuh"
#include "qcu_blas_public.h"
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_complex_xsay.h"

namespace qcu::qcu_blas {

template <typename _Float>
using Complex_xsayArgument = typename qcu::qcu_blas::Complex_xsay<_Float>::Complex_xsayArgument;

template <typename _Float>
void Complex_xsay<_Float, std::enable_if_t <std::is_same_v<_Float, float>  ||
                                            std::is_same_v<_Float, double> ||
                                            std::is_same_v<_Float, half> > >
    ::operator()(Complex_xsayArgument arg) 
{
  int threads_per_block = std::min(512, maxThreadsPerBlock);
  int blocks_per_grid   = std::min(div_ceil(arg.single_vec_len, threads_per_block),
                                              maxGridSize);

  int total_vecs = arg.inc_idx;
  for (int i = 0; i < total_vecs; ++i) {
    qcu::device::kernel::cxsay_stride_kernel<_Float> 
          <<<blocks_per_grid, threads_per_block>>>
              (arg.res, arg.x, arg.a, arg.y, arg.single_vec_len, arg.inc_idx, i);  
  }
}

template struct Complex_xsay<half>;
template struct Complex_xsay<float>;
template struct Complex_xsay<double>;
}