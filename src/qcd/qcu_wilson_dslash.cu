#include <cuda_fp16.h>

#include "kernel/su_n_m_rhs_dslash.cuh"
#include "qcd/qcu_dslash.h"
#include "qcu_public.h"
#include "qcu_wmma_constant.h"
#include "check_error/check_cuda.cuh"
namespace qcu {

// clang-format off
template <typename Float>
inline void ApplyWilsonDslash_Mrhs( Float* __restrict__ out, Float* __restrict__ in, Float* __restrict__ gauge, 
                                    int Lx, int Ly, int Lz, int Lt, int g_x, int g_y, int g_z, int g_t, 
                                    int parity, bool dagger_flag, int n_color, int m_rhs, cudaStream_t& stream
) {
    // clang-format on
    int vol = Lx * Ly * Lz * Lt / 2;
    int warp_num_per_block = WARP_PER_BLOCK;

    dim3 block_size(WARP_SIZE, warp_num_per_block);
    dim3 grid_size(vol);
    device::wilson_dslash_su_n_mrhs<Float> <<<grid_size, block_size, 0, stream>>>(
        out, in, gauge, Lx, Ly, Lz, Lt, g_x, g_y, g_z, g_t, parity, dagger_flag, n_color, m_rhs);
}

void WilsonDslash::apply() {
    int Lx = dslashParam_->lattDesc->dims[X_DIM];
    int Ly = dslashParam_->lattDesc->dims[Y_DIM];
    int Lz = dslashParam_->lattDesc->dims[Z_DIM];
    int Lt = dslashParam_->lattDesc->dims[T_DIM];

    int g_x = dslashParam_->procDesc->dims[X_DIM];
    int g_y = dslashParam_->procDesc->dims[Y_DIM];
    int g_z = dslashParam_->procDesc->dims[Z_DIM];
    int g_t = dslashParam_->procDesc->dims[T_DIM];

    // clang-format off
    switch (dslashParam_->precision) {
        case QCU_HALF_PRECISION:
            ApplyWilsonDslash_Mrhs<half>(static_cast<half*>(dslashParam_->fermionOut_MRHS), 
                                         static_cast<half*>(dslashParam_->fermionIn_MRHS), 
                                         static_cast<half*>(dslashParam_->gauge), 
                                         Lx, Ly, Lz, Lt, g_x, g_y, g_z, g_t, 
                                         dslashParam_->parity, dslashParam_->daggerFlag, 
                                         dslashParam_->nColor, dslashParam_->mInput, 
                                         dslashParam_->stream1);
            /* code */
            break;
        case QCU_SINGLE_PRECISION:
            errorQcu("Not implemented yet\n");  // TODO
            assert(0);
            break;
        case QCU_DOUBLE_PRECISION:
            ApplyWilsonDslash_Mrhs<double>(static_cast<double*>(dslashParam_->fermionOut_MRHS), 
                                           static_cast<double*>(dslashParam_->fermionIn_MRHS), 
                                           static_cast<double*>(dslashParam_->gauge), 
                                           Lx, Ly, Lz, Lt, g_x, g_y, g_z, g_t, 
                                           dslashParam_->parity, dslashParam_->daggerFlag, 
                                           dslashParam_->nColor, dslashParam_->mInput, 
                                           dslashParam_->stream1); 
            break;
        default:
            errorQcu("Not implemented yet\n");  // TODO
            assert(0);
            break;
    }
    CHECK_CUDA(cudaStreamSynchronize(dslashParam_->stream1));
    // clang-format on
}
void WilsonDslash::preApply() {
    errorQcu("Not implemented yet\n");  // TODO
}
void WilsonDslash::postApply() {
    errorQcu("Not implemented yet\n");  // TODO
}
// TODO : calc flops
void WilsonDslash::flops() {
    errorQcu("Not implemented yet\n");  // TODO
}

}  // namespace qcu