#include "kernel/element_wise/c_axpby.cuh"
// #include "qcu_blas/qcu_blas.h"
#include "qcu_blas_public.h"
#include "qcu_utils.h"
#include "qcu_blas/qcu_blas_complex_axpby.h"

namespace qcu::qcu_blas {

template <typename ComputeFloat_, typename ScaleFloat_ = ComputeFloat_>
using Complex_axpbyArgument = typename qcu::qcu_blas::Complex_axpby<ComputeFloat_>::Complex_axpbyArgument;

template <typename ComputeFloat_, typename ScaleFloat_>
void Complex_axpby<
    ComputeFloat_,
    ScaleFloat_,
    std::enable_if_t <std::is_same_v<ComputeFloat_, float>  || std::is_same_v<ComputeFloat_, double> || std::is_same_v<ComputeFloat_, half> > >
    ::operator()(Complex_axpbyArgument arg) 
{
    int threads_per_block = std::min(512, maxThreadsPerBlock);
    int blocks_per_grid   = std::min(div_ceil(arg.single_vec_len, threads_per_block), maxGridSize);

    int total_vecs = arg.inc_idx;
    for (int i = 0; i < total_vecs; ++i) {
        qcu::device::kernel::caxpby_stride_kernel<ComputeFloat_, ScaleFloat_> <<<blocks_per_grid, threads_per_block>>>
            (arg.res, arg.a, arg.x, arg.b, arg.y, arg.single_vec_len, arg.inc_idx, i);
    }
}

template struct Complex_axpby<half, half>;
template struct Complex_axpby<half, float>;
template struct Complex_axpby<float, float>;
template struct Complex_axpby<double, double>;
}