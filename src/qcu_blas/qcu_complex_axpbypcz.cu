#include "kernel/element_wise/c_axpbypcz.cuh"
#include "qcu_blas/qcu_blas_complex_axpbypcz.h"
#include "qcu_blas_public.h"
#include "qcu_utils.h"

namespace qcu::qcu_blas {

template <typename _Float>
using Complex_axpbypczArgument = typename qcu::qcu_blas::Complex_axpbypcz<_Float>::Complex_axpbypczArgument;

template <typename _Float>
void Complex_axpbypcz<_Float, std::enable_if_t <std::is_same_v<_Float, float>  ||
                                                std::is_same_v<_Float, double> ||
                                                std::is_same_v<_Float, half> > >
    ::operator()(Complex_axpbypczArgument arg) 
{
  int threads_per_block = std::min(512, maxThreadsPerBlock);
  int blocks_per_grid   = std::min(div_ceil(arg.single_vec_len, threads_per_block),
                                              maxGridSize);

  int total_vecs = arg.inc_idx;
  for (int i = 0; i < total_vecs; ++i) {
    qcu::device::kernel::caxpbypcz_stride_kernel<_Float> <<<blocks_per_grid, threads_per_block>>>
              (arg.res, arg.a, arg.x, arg.b, arg.y, arg.c, arg.z, arg.single_vec_len, arg.inc_idx, i);  
  }
}

template struct Complex_axpbypcz<half>;
template struct Complex_axpbypcz<float>;
template struct Complex_axpbypcz<double>;
}