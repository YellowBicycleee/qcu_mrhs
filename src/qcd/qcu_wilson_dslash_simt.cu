#include <cuda_fp16.h>

#include "check_error/check_cuda.cuh"
#include "cuda_utils.cuh"
#include "kernel/gemm/qcu_gemm_configure.cuh"
#include "kernel/sun_mrhs_wilson_dslash_simt_pack.cuh"
#include "kernel/sun_mrhs_wilson_dslash_simt.cuh"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_base/qcu_alloc.h"
#include "qcu_config/qcu_config.h"
#include "qcu_public.h"
#include "qcu_utils.h"

namespace qcu::simt {

template <typename Float>
inline void ApplyWilsonDslash_Mrhs( DslashParam& dslash_param)
{
    int half_vol = config::lattice_volume_local() / 2;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    using BlockShape = gemm::GemmShape<8, 8, 8>;
    // using BlockShape = gemm::GemmShape<16, 16, 16>;
    unsigned int multiprocess_mask = config::get_mpi_separated_mask();

    int blk_x = BlockShape::kN;
    int blk_y = BlockShape::kM;

    dim3 grid_size(div_ceil(dslash_param.n_color, blk_x), div_ceil(dslash_param.m_input, blk_y), std::min(half_vol, 65535));
    dim3 block_size(blk_x, blk_y, 1);

    // printf("SIMT dslash Beginning\n");
    qcu::device::wilson_dslash_su_n_mrhs<Float, BlockShape>
        <<<grid_size, block_size, 0, dslash_param.streams[8]>>>
        (   static_cast<Float*>(dslash_param.fermion_out_MRHS),
            static_cast<Float*>(dslash_param.fermion_in_MRHS),
            static_cast<Float*>(dslash_param.gauge),
            latt_desc, multiprocess_mask,
            dslash_param.parity, dslash_param.dagger_flag,
            dslash_param.n_color, dslash_param.m_input);

    CHECK_CUDA(cudaDeviceSynchronize());
    // printf("SIMT dslash Ending, config = grid(%d, %d, %d), block(%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
}

void WilsonDslash::apply(std::shared_ptr<DslashParam> dslash_param) {
    timer_.reset();
    timer_.start();

    pre_apply(dslash_param);

    int m_input = dslash_param->m_input;
    int n_color = dslash_param->n_color;
    int half_vol = config::lattice_volume_local() / 2;

    flop_per_rhs_ = (128 * n_color * n_color + 96 * n_color) * half_vol;
    flop_ += flop_per_rhs_ * m_input;

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
                errorQcu("Not implemented yet\n");
                assert(0);
            }
            break;
    }
    CHECK_CUDA(cudaStreamSynchronize(dslash_param->streams[8]));
    post_apply(dslash_param);

    double time = timer_.lap_ms();
    time_ += time;
}

void WilsonDslash::post_apply(const std::shared_ptr<DslashParam> dslash_param) {
    for (int i = 0; i < Nd; ++i) {
        if (dslash_param->proc_desc->at(i) > 1) {
            apply_ghost_unpack(*dslash_param, i);
        }
    }
}

}