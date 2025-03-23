#include "kernel/element_wise/c_xsay.cuh"
#include "qcu_blas_public.h"
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_complex_xsay.h"

namespace qcu::qcu_blas {

template <typename ComputeFloat_, typename ScaleFloat_>
using Complex_xsayArgument = typename qcu::qcu_blas::Complex_xsay<ComputeFloat_, ScaleFloat_>::Complex_xsayArgument;

template <typename ComputeFloat_, typename ScaleFloat_>
void Complex_xsay<ComputeFloat_,ScaleFloat_,
    std::enable_if_t <std::is_same_v<ComputeFloat_, float> || std::is_same_v<ComputeFloat_, double> || std::is_same_v<ComputeFloat_, half> > >
    ::operator()(Complex_xsayArgument arg)
{
    int threads_per_block = std::min(512, maxThreadsPerBlock);
    int blocks_per_grid   = std::min(div_ceil(arg.single_vec_len, threads_per_block), maxGridSize);

    int total_vecs = arg.inc_idx;
    for (int i = 0; i < total_vecs; ++i) {
        qcu::device::kernel::cxsay_stride_kernel<ComputeFloat_, ScaleFloat_>
            <<<blocks_per_grid, threads_per_block>>>
            (arg.res, arg.x, arg.a, arg.y, arg.single_vec_len, arg.inc_idx, i);
    }
}

template struct Complex_xsay<half, half>;
template struct Complex_xsay<half, float>;
template struct Complex_xsay<float, float>;
template struct Complex_xsay<double, double>;
}