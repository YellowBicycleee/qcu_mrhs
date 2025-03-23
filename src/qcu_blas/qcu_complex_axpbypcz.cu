#include "kernel/element_wise/c_axpbypcz.cuh"
#include "qcu_blas/qcu_blas_complex_axpbypcz.h"
#include "qcu_blas_public.h"
#include "qcu_utils.h"

namespace qcu::qcu_blas {

template <typename ComputeFloat_, typename ScaleFloat_>
using Complex_axpbypczArgument
    = typename qcu::qcu_blas::Complex_axpbypcz<ComputeFloat_, ScaleFloat_>::Complex_axpbypczArgument;

template <typename ComputeFloat_, typename ScaleFloat_>
void Complex_axpbypcz<
    ComputeFloat_,
    ScaleFloat_,
    std::enable_if_t <std::is_same_v<ComputeFloat_, float> || std::is_same_v<ComputeFloat_, double> || std::is_same_v<ComputeFloat_, half> > >
    ::operator()(Complex_axpbypczArgument arg) 
{
    int threads_per_block = std::min(512, maxThreadsPerBlock);
    int blocks_per_grid   = std::min(div_ceil(arg.single_vec_len, threads_per_block), maxGridSize);

    int total_vecs = arg.inc_idx;
    for (int i = 0; i < total_vecs; ++i) {
    qcu::device::kernel::caxpbypcz_stride_kernel<ComputeFloat_, ScaleFloat_>
        <<<blocks_per_grid, threads_per_block>>>
        (arg.res, arg.a, arg.x, arg.b, arg.y, arg.c, arg.z, arg.single_vec_len, arg.inc_idx, i);
    }
}

template struct Complex_axpbypcz<half, half>;
template struct Complex_axpbypcz<half, float>;
template struct Complex_axpbypcz<float, float>;
template struct Complex_axpbypcz<double, double>;
}