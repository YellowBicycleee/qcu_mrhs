#include <cuda_fp16.h>

// #ifdef QCU_ARCH_WMMA_SM80_ENABLED
#include "kernel/su_n_m_rhs_dslash.cuh"
// #endif // QCU_ARCH_WMMA_SM80_ENABLED

#include "qcd/qcu_dslash_wilson.h"
#include "qcu_public.h"
#include "check_error/check_cuda.cuh"
#include "qcu_config/qcu_config.h"

namespace qcu {

// clang-format off
template <typename Float>
inline void ApplyWilsonDslash_Mrhs( DslashParam& dslash_param)
{

// #ifdef QCU_ARCH_WMMA_SM80_ENABLED
    int half_vol = config::lattice_volume_local() / 2;
    int warp_num_per_block = WARP_PER_BLOCK;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);
    const qcu::QcuProcDesc& proc_desc = *(dslash_param.proc_desc);
    dim3 block_size(WARP_SIZE, warp_num_per_block);
    dim3 grid_size(half_vol);
    qcu::device::wilson_dslash_su_n_mrhs<Float> <<<grid_size, block_size, 0, dslash_param.stream1>>>(
        static_cast<Float*>(dslash_param.fermion_out_MRHS),
        static_cast<Float*>(dslash_param.fermion_in_MRHS),
        static_cast<Float*>(dslash_param.gauge),
        latt_desc.X(), latt_desc.Y(), latt_desc.Z(), latt_desc.T(),
        proc_desc.X(), proc_desc.Y(), proc_desc.Z(), proc_desc.T(),
        dslash_param.parity, dslash_param.dagger_flag, dslash_param.n_color, dslash_param.m_input);
}

void WilsonDslash::apply(std::shared_ptr<DslashParam> dslash_param) {

    // clang-format off
    switch (dslash_param->dslash_precision) {
        case QcuPrecision::kPrecisionHalf:
            { ApplyWilsonDslash_Mrhs<half>(*dslash_param); }
            break;
        case QcuPrecision::kPrecisionSingle:
            {
                errorQcu("Not implemented yet\n");  // TODO
            }
            break;
        case QcuPrecision::kPrecisionDouble:
            { ApplyWilsonDslash_Mrhs<double>(*dslash_param);}
            break;
        default:
            {
                errorQcu("Not implemented yet\n");  // TODO
            }
            break;
    }
    CHECK_CUDA(cudaStreamSynchronize(dslash_param->stream1));
    // clang-format on
}
void WilsonDslash::pre_apply(const std::shared_ptr<DslashParam> dslash_param) {
    errorQcu("Not implemented yet\n");  // TODO
}
void WilsonDslash::post_apply(const std::shared_ptr<DslashParam> dslash_param) {
    errorQcu("Not implemented yet\n");  // TODO
}
// TODO : calc flops
double WilsonDslash::flops() {
    errorQcu("Not implemented yet\n");  // TODO
}

}  // namespace qcu