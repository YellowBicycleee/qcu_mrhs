#include <cuda_fp16.h>

#include "kernel/su_n_m_rhs_dslash.cuh"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_public.h"
#include "check_error/check_cuda.cuh"
#include "qcu_config/qcu_config.h"

namespace qcu {

// clang-format off
template <typename Float>
inline void ApplyWilsonDslash_Mrhs( DslashParam& dslash_param)
{
    // clang-format on
    int half_vol = config::lattice_volume() / 2;
    int warp_num_per_block = WARP_PER_BLOCK;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.lattDesc);
    const qcu::QcuProcDesc& proc_desc = *(dslash_param.procDesc);
    dim3 block_size(WARP_SIZE, warp_num_per_block);
    dim3 grid_size(half_vol);
    device::wilson_dslash_su_n_mrhs<Float> <<<grid_size, block_size, 0, dslash_param.stream1>>>(
        static_cast<Float*>(dslash_param.fermionOut_MRHS),
        static_cast<Float*>(dslash_param.fermionIn_MRHS),
        static_cast<Float*>(dslash_param.gauge),
        latt_desc.X(), latt_desc.Y(), latt_desc.Z(), latt_desc.T(),
        proc_desc.X(), proc_desc.Y(), proc_desc.Z(), proc_desc.T(),
        dslash_param.parity, dslash_param.daggerFlag, dslash_param.nColor, dslash_param.mInput);
}

void WilsonDslash::apply(std::shared_ptr<DslashParam> dslash_param) {

    // clang-format off
    switch (dslash_param->precision) {
        case QcuPrecision::kPrecisionHalf:
            { ApplyWilsonDslash_Mrhs<half>(*dslash_param); }
            break;
        case QcuPrecision::kPrecisionSingle:
            {
                errorQcu("Not implemented yet\n");  // TODO
                assert(0);
            }
            break;
        case QcuPrecision::kPrecisionDouble:
            { ApplyWilsonDslash_Mrhs<double>(*dslash_param);}
            break;
        default:
            {
                errorQcu("Not implemented yet\n");  // TODO
                assert(0);
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
    if (if_metric_) {
        return operations_ / time_;
    } else {
        return 0;
    }
}

}  // namespace qcu