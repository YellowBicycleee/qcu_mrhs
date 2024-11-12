#include <cuda_fp16.h>

#include "kernel/su_n_m_rhs_dslash_new.cuh"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_public.h"
#include "check_error/check_cuda.cuh"

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

namespace qcu {
namespace developing {
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

void WilsonDslash::apply(DslashParam& dslashParam) {
    errorQcu("Not implemented yet\n");  // TODO

    // int Lx = dslashParam.lattDesc->data[X_DIM];
    // int Ly = dslashParam.lattDesc->data[Y_DIM];
    // int Lz = dslashParam.lattDesc->data[Z_DIM];
    // int Lt = dslashParam.lattDesc->data[T_DIM];
    //
    // int g_x = dslashParam.procDesc->data[X_DIM];
    // int g_y = dslashParam.procDesc->data[Y_DIM];
    // int g_z = dslashParam.procDesc->data[Z_DIM];
    // int g_t = dslashParam.procDesc->data[T_DIM];
    //
    // // clang-format off
    // switch (dslashParam.precision) {
    //     case QcuPrecision::kPrecisionHalf:
    //         ApplyWilsonDslash_Mrhs<half>(static_cast<half*>(dslashParam.fermionOut_MRHS),
    //                                      static_cast<half*>(dslashParam.fermionIn_MRHS),
    //                                      static_cast<half*>(dslashParam.gauge),
    //                                      Lx, Ly, Lz, Lt, g_x, g_y, g_z, g_t,
    //                                      dslashParam.parity, dslashParam.daggerFlag,
    //                                      dslashParam.nColor, dslashParam.mInput,
    //                                      dslashParam.stream1);
    //         break;
    //     case QcuPrecision::kPrecisionSingle:
    //         errorQcu("Not implemented yet\n");  // TODO
    //         assert(0);
    //         break;
    //     case QcuPrecision::kPrecisionDouble:
    //         ApplyWilsonDslash_Mrhs<double>(static_cast<double*>(dslashParam.fermionOut_MRHS),
    //                                        static_cast<double*>(dslashParam.fermionIn_MRHS),
    //                                        static_cast<double*>(dslashParam.gauge),
    //                                        Lx, Ly, Lz, Lt, g_x, g_y, g_z, g_t,
    //                                        dslashParam.parity, dslashParam.daggerFlag,
    //                                        dslashParam.nColor, dslashParam.mInput,
    //                                        dslashParam.stream1);
    //         break;
    //     default:
    //         errorQcu("Not implemented yet\n");  // TODO
    //         assert(0);
    //         break;
    // }
    // CHECK_CUDA(cudaStreamSynchronize(dslashParam.stream1));
    // clang-format on
}
void WilsonDslash::pre_apply(const DslashParam&) {
    errorQcu("Not implemented yet\n");  // TODO
}
void WilsonDslash::post_apply(const DslashParam&) {
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
}
}  // namespace qcu