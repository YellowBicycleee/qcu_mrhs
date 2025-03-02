#include "qcd/qcu_dslash_staggered.h"
#include "kernel/sun_mrhs_staggered_dslash_simt.cuh"
#include "qcu_config/qcu_config.h"
#include "qcu_utils.h"

namespace qcu::simt {

template <typename Float>
inline void ApplyStaggeredDslash_Mrhs(DslashParam& dslash_param)
{
    int half_vol = config::lattice_volume_local() / 2;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    using BlockShape = gemm::GemmShape<8, 8, 8>;
    // using BlockShape = gemm::GemmShape<16, 16, 16>;
    unsigned int multiprocess_mask = config::get_mpi_separated_mask();

    int blk_x = BlockShape::kM;
    int blk_y = BlockShape::kN;

    dim3 grid_size(div_ceil(dslash_param.n_color, blk_x), div_ceil(dslash_param.m_input, blk_y), std::min(half_vol, 65535));
    dim3 block_size(blk_x, blk_y, 1);

    qcu::device::staggered_dslash_su_n_mrhs <Float, BlockShape>
        <<<grid_size, block_size, 0, dslash_param.stream1>>> 
    (
        static_cast<Float*>(dslash_param.fermion_out_MRHS),
        static_cast<Float*>(dslash_param.fermion_in_MRHS),
        static_cast<Float*>(dslash_param.gauge),
        latt_desc, multiprocess_mask,
        dslash_param.parity, dslash_param.dagger_flag,
        dslash_param.n_color, dslash_param.m_input,
        /* kappa*/ 0, /* mat_flag */ false,
        /* t_boundary */ 1, /* staggered_phase */ QcuStaggeredPhase::kQcuStaggeredPhaseNo
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

void StaggeredDslash::apply(const std::shared_ptr<DslashParam> dslash_param) {

    int m_input = dslash_param->m_input;
    int n_color = dslash_param->n_color;
    int half_vol = config::lattice_volume_local() / 2;
    double num_operations = static_cast<double>(half_vol * m_input * (
        2 * Nd * Nspin_ * n_color   // project
        + 2 * Nd * Nspin_ / 2 * (8 * n_color  - 2) * n_color  // GEMV
        + (2 * Nd - 1) * Nspin_ * n_color  // reconstruct
    ));
    operations_cur_ += num_operations;
    operations_total_ += num_operations;

    switch (dslash_param->dslash_precision) {
        case QcuPrecision::kPrecisionHalf:
        { ApplyStaggeredDslash_Mrhs<half>(*dslash_param); }
        break;
        case QcuPrecision::kPrecisionSingle:
        { ApplyStaggeredDslash_Mrhs<float>(*dslash_param); }
        break;
        case QcuPrecision::kPrecisionDouble:
        { ApplyStaggeredDslash_Mrhs<double>(*dslash_param);}
        break;
        default:
        {
            errorQcu("Not implemented yet\n");
            assert(0);
        }
        break;
    }
    CHECK_CUDA(cudaStreamSynchronize(dslash_param->stream1));
    // post_apply(dslash_param);
}

double StaggeredDslash::flops() {
    return 0.0;
}

void StaggeredDslash::pre_apply(const std::shared_ptr<DslashParam>) {
    errorQcu("Not implemented yet\n");
}

void StaggeredDslash::post_apply(const std::shared_ptr<DslashParam>) {
    errorQcu("Not implemented yet\n");
}

void StaggeredDslash::apply_ghost_unpack(DslashParam& dslash_param, int ghost_dim) {
    errorQcu("Not implemented yet\n");
}

void StaggeredDslash::apply_ghost_pack(DslashParam& dslash_param, int ghost_dim) {
    errorQcu("Not implemented yet\n");
}

}
