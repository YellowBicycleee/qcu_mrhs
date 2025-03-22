#include <mpi.h>
#include "check_error/check_cuda.cuh"
#include "check_error/check_mpi.h"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_config/qcu_config.h"

#ifdef COMPILE_TENSOR_CORE_CODE
#include "kernel/sun_mrhs_wilson_dslash_tensorop_pack.cuh"
#endif
namespace qcu::tensorop {
// pack 使用独立的8个流
template <typename Float_>
inline void apply_sun_mrhs_dslash_ghost_pack (DslashParam& dslash_param, int ghost_dim) {
#ifdef COMPILE_TENSOR_CORE_CODE
    unsigned int multiprocess = config::get_mpi_separated_mask();

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    int half_vol = config::lattice_volume_local() / 2;
    int num_threads = half_vol / latt_desc.at(ghost_dim);

    void* fwd_pack_buf = dslash_param.fermion_ghost->get_pack_buf_at(ghost_dim, FWD);
    void* bwd_pack_buf = dslash_param.fermion_ghost->get_pack_buf_at(ghost_dim, BWD);

    cudaStream_t fwd_stream = dslash_param.streams[ghost_dim * 2 + FWD];
    cudaStream_t bwd_stream = dslash_param.streams[ghost_dim * 2 + BWD];

    if constexpr (std::is_same_v<Float_, double>) {
        using BlockShape = gemm::GemmShape<8, 8, 4>;
        using WarpShape = gemm::GemmShape<8, 8, 4>;
        // dim3 block_size(BlockShape::kN /2 * BlockShape::kM / 2);
        dim3 block_size(32 * BlockShape::kMN / WarpShape::kMN);
        dim3 grid_size(div_ceil(dslash_param.m_input, BlockShape::kN),
            div_ceil(dslash_param.n_color, BlockShape::kM),
            std::min(half_vol, 65535));
        qcu::device::tensorop::wilson_dslash_sun_mrhs_forward_ghost_pack
        <Float_, BlockShape, WarpShape>
            <<<grid_size, block_size, 0, fwd_stream>>> (
                static_cast<Float_*>(fwd_pack_buf),
                static_cast<Float_*>(dslash_param.fermion_in_MRHS),
                static_cast<Float_*>(dslash_param.gauge),
                ghost_dim, latt_desc, multiprocess,
                dslash_param.parity, dslash_param.dagger_flag,
                dslash_param.n_color, dslash_param.m_input);

        qcu::device::tensorop::wilson_dslash_sun_mrhs_backward_ghost_pack
        <Float_, BlockShape, WarpShape>
            <<<grid_size, block_size, 0, bwd_stream>>> (
                static_cast<Float_*>(bwd_pack_buf),
                static_cast<Float_*>(dslash_param.fermion_in_MRHS),
                static_cast<Float_*>(dslash_param.gauge),
                ghost_dim, latt_desc, multiprocess,
                dslash_param.parity, dslash_param.dagger_flag,
                dslash_param.n_color, dslash_param.m_input);
    }
    else if constexpr (std::is_same_v<Float_, half>) {
        using BlockShape = gemm::GemmShape<16, 16, 16>;
        using WarpShape = gemm::GemmShape<16, 16, 16>;
        dim3 block_size(32 * BlockShape::kMN / WarpShape::kMN);
        dim3 grid_size(div_ceil(dslash_param.m_input, BlockShape::kN),
            div_ceil(dslash_param.n_color, BlockShape::kM),
            std::min(half_vol, 65535));
        qcu::device::tensorop::wilson_dslash_sun_mrhs_forward_ghost_pack
        <Float_, BlockShape, WarpShape>
            <<<grid_size, block_size, 0, fwd_stream>>> (
                static_cast<Float_*>(fwd_pack_buf), static_cast<Float_*>(dslash_param.fermion_in_MRHS),
                static_cast<Float_*>(dslash_param.gauge), ghost_dim, latt_desc,
                multiprocess,
                dslash_param.parity, dslash_param.dagger_flag,
                dslash_param.n_color, dslash_param.m_input);

        qcu::device::tensorop::wilson_dslash_sun_mrhs_backward_ghost_pack
        <Float_, BlockShape, WarpShape>
            <<<grid_size, block_size, 0, bwd_stream>>> (
                static_cast<Float_*>(bwd_pack_buf), static_cast<Float_*>(dslash_param.fermion_in_MRHS),
                static_cast<Float_*>(dslash_param.gauge), ghost_dim, latt_desc,
                multiprocess,
                dslash_param.parity, dslash_param.dagger_flag,
                dslash_param.n_color, dslash_param.m_input);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
#else
    errorQcu("TensorOp not supported\n");
#endif
}

void WilsonDslash::apply_ghost_pack(DslashParam& dslash_param, int ghost_dim) {

    switch (dslash_param.dslash_precision) {
        case QcuPrecision::kPrecisionHalf:
            {   apply_sun_mrhs_dslash_ghost_pack<half>(dslash_param, ghost_dim);    }
            break;
        case QcuPrecision::kPrecisionSingle:
            {   apply_sun_mrhs_dslash_ghost_pack<float>(dslash_param, ghost_dim);   }
            break;
        case QcuPrecision::kPrecisionDouble:
            {   apply_sun_mrhs_dslash_ghost_pack<double>(dslash_param, ghost_dim); }
            break;
        default:
            {   errorQcu("Not implemented yet\n"); }
            break;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    QcuProcDesc proc_desc = *(dslash_param.proc_desc);
    qcu::FourDimCoordinate mpi_coord_forward = config::get_mpi_coord();
    mpi_coord_forward.data[ghost_dim] = (mpi_coord_forward.data[ghost_dim] + 1) % proc_desc.data[ghost_dim];

    qcu::FourDimCoordinate mpi_coord_backward = config::get_mpi_coord();
    mpi_coord_backward.data[ghost_dim] = (mpi_coord_backward.data[ghost_dim] - 1 + proc_desc.data[ghost_dim]) % proc_desc.data[ghost_dim];

    size_t type_size = 0;
    switch (dslash_param.dslash_precision) {
        case kPrecisionDouble:
            type_size = sizeof(double) * 2;
            break;
        case kPrecisionSingle:
            type_size = sizeof(float) * 2;
            break;
        case kPrecisionHalf:
            type_size = sizeof(half) * 2;
            break;
        default:
            type_size = 0;
            break;
    }

    FourDimDesc mpi_desc {proc_desc.data[X_DIM], proc_desc.data[Y_DIM],proc_desc.data[Z_DIM], proc_desc.data[T_DIM]};

    const int byte_size = dslash_param.fermion_ghost->ghost_len[ghost_dim] * type_size;
    void* device_pack_buf_fwd = dslash_param.fermion_ghost->get_pack_buf_at(ghost_dim, FWD);
    void* host_pack_buf_fwd = dslash_param.fermion_ghost->get_host_pack_buf_at(ghost_dim, FWD);

    CHECK_CUDA(cudaMemcpy(host_pack_buf_fwd, device_pack_buf_fwd, byte_size, cudaMemcpyDeviceToHost));

    CHECK_MPI(
        MPI_Isend(host_pack_buf_fwd, byte_size, MPI_BYTE, mpi_coord_forward.getReversedIdx1D(mpi_desc),
            BWD, MPI_COMM_WORLD, &config::get_mpi_request_pack(ghost_dim, FWD))
    );


    void* device_pack_buf_bwd = dslash_param.fermion_ghost->get_pack_buf_at(ghost_dim, BWD);
    void* host_pack_buf_bwd = dslash_param.fermion_ghost->get_host_pack_buf_at(ghost_dim, BWD);
    CHECK_CUDA(cudaMemcpy(host_pack_buf_bwd, device_pack_buf_bwd, byte_size, cudaMemcpyDeviceToHost));

    CHECK_MPI(
        MPI_Isend(host_pack_buf_bwd, byte_size, MPI_BYTE, mpi_coord_backward.getReversedIdx1D(mpi_desc),
            FWD, MPI_COMM_WORLD, &config::get_mpi_request_pack(ghost_dim, BWD))
        );

}


}
