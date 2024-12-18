#include <cuda_fp16.h>

#include "check_error/check_cuda.cuh"
#include "kernel/sun_mrhs_wilson_dslash_simt.cuh"
#include "kernel/sun_mrhs_wilson_dslash_ghost_simt.cuh"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_base/qcu_alloc.h"
#include "qcu_config/qcu_config.h"
#include "qcu_public.h"
#include "cuda_utils.cuh"

#include "qcu_base/qcu_base.h"
/** how to impl dslash
 * Policy1: MPI Naive
 *    main thread-flag=false----------stream9--------------------------------------internal Kernel----flag=true--------join, sync---flag=false
 *                          |----sub_thread1(stream1)---EXTERNAL KERNEL---D2H---MPI---H2D----MPI_Test--|---->stream9------|
 *                          |----sub_thread2(stream2)---EXTERNAL KERNEL---D2H---MPI---H2D----MPI_Test--|---->stream9------|
 *                          |----sub_thread3(stream3)---EXTERNAL KERNEL---D2H---MPI---H2D----MPI_Test--|---->stream9------|
 *                          |----sub_thread4(stream4)---EXTERNAL KERNEL---D2H---MPI---H2D----MPI_Test--|---->stream9------|
 *                          |----sub_thread5(stream5)---EXTERNAL KERNEL---D2H---MPI---H2D----MPI_Test--|---->stream9------|
 *                          |----sub_thread6(stream6)---EXTERNAL KERNEL---D2H---MPI---H2D----MPI_Test--|---->stream9------|
 *                          |----sub_thread7(stream7)---EXTERNAL KERNEL---D2H---MPI---H2D----MPI_Test--|---->stream9------|
 *                          |----sub_thread8(stream8)---EXTERNAL KERNEL---D2H---MPI---H2D----MPI_Test--|---->stream9------|
 *                          |---------------------------------------------------------------------------------------------|
 * Policy2: MPI + NCCL
 */

namespace qcu::developing {
// template <typename Float, typename CudaArch = qcu::arch>
// inline void ApplyWilsonDslash_Mrhs( DslashParam& dslash_param);

template <typename Float>
inline void ApplyWilsonDslash_Mrhs( DslashParam& dslash_param)
{
    int half_vol = config::lattice_volume_local() / 2;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);
    const qcu::QcuProcDesc& proc_desc = *(dslash_param.proc_desc);

    using BlockShape = gemm::GemmShape<8, 8, 8>;
    // using BlockShape = gemm::GemmShape<16, 16, 16>;
    int multiprocess = 0;
    for (int i = 0; i < Nd; ++i) {
        if (proc_desc.data[i] > 0) {
            multiprocess |= (1 << i);
        }
    }

    int blk_x = BlockShape::kM;
    int blk_y = BlockShape::kN;

    dim3 grid_size(div_ceil(dslash_param.n_color, blk_x), div_ceil(dslash_param.m_input, blk_y), min(half_vol, 65535));
    dim3 block_size(blk_x, blk_y, 1);

    printf("SIMT dslash Beginning\n");
    qcu::device::wilson_dslash_su_n_mrhs<Float, BlockShape>
        <<<grid_size, block_size, 0, dslash_param.stream1>>>
        (   static_cast<Float*>(dslash_param.fermion_out_MRHS),
            static_cast<Float*>(dslash_param.fermion_in_MRHS),
            static_cast<Float*>(dslash_param.gauge),
            latt_desc, multiprocess,
            dslash_param.parity, dslash_param.dagger_flag,
            dslash_param.n_color, dslash_param.m_input);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("SIMT dslash Ending, config = grid(%d, %d, %d), block(%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
}

void WilsonDslash::apply(std::shared_ptr<DslashParam> dslash_param) {

    // clang-format off
    switch (dslash_param->dslash_precision) {
        case QcuPrecision::kPrecisionHalf:
            { ApplyWilsonDslash_Mrhs<half>(*dslash_param); }
            break;
        case QcuPrecision::kPrecisionSingle:
            { ApplyWilsonDslash_Mrhs<float>(*dslash_param); }
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

}