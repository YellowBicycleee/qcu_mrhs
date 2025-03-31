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

    unsigned int multiprocess = config::get_mpi_separated_mask();

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    int half_vol = config::lattice_volume_local() / 2;
    int num_threads = half_vol / latt_desc.at(ghost_dim);

    void* d_fwd_pack_buf = dslash_param.fermion_ghost->get_pack_buf_at(ghost_dim, FWD);
    void* d_bwd_pack_buf = dslash_param.fermion_ghost->get_pack_buf_at(ghost_dim, BWD);

    cudaStream_t fwd_stream = dslash_param.streams[ghost_dim * 2 + FWD];
    cudaStream_t bwd_stream = dslash_param.streams[ghost_dim * 2 + BWD];

    if constexpr (std::is_same_v<Float_, double>) {
        using BlockShape = gemm::GemmShape<8, 8, 4>;
        using WarpShape = gemm::GemmShape<8, 8, 4>;
        // dim3 block_size(BlockShape::kN /2 * BlockShape::kM / 2);
        dim3 block_size(32 * BlockShape::kMN / WarpShape::kMN);
        dim3 grid_size(div_ceil(dslash_param.m_input, BlockShape::kN),
            div_ceil(dslash_param.n_color, BlockShape::kM),
            std::min(num_threads, 65535));
        qcu::device::tensorop::wilson_dslash_sun_mrhs_forward_ghost_pack
        <Float_, BlockShape, WarpShape>
            <<<grid_size, block_size, 0, fwd_stream>>> (
                static_cast<Float_*>(d_fwd_pack_buf),
                static_cast<Float_*>(dslash_param.fermion_in_MRHS),
                static_cast<Float_*>(dslash_param.gauge),
                ghost_dim, latt_desc, multiprocess,
                dslash_param.parity, dslash_param.dagger_flag,
                dslash_param.n_color, dslash_param.m_input);

        qcu::device::tensorop::wilson_dslash_sun_mrhs_backward_ghost_pack
        <Float_, BlockShape, WarpShape>
            <<<grid_size, block_size, 0, bwd_stream>>> (
                static_cast<Float_*>(d_bwd_pack_buf),
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
            std::min(num_threads, 65535));
        qcu::device::tensorop::wilson_dslash_sun_mrhs_forward_ghost_pack
        <Float_, BlockShape, WarpShape>
            <<<grid_size, block_size, 0, fwd_stream>>> (
                static_cast<Float_*>(d_fwd_pack_buf), static_cast<Float_*>(dslash_param.fermion_in_MRHS),
                static_cast<Float_*>(dslash_param.gauge), ghost_dim, latt_desc,
                multiprocess,
                dslash_param.parity, dslash_param.dagger_flag,
                dslash_param.n_color, dslash_param.m_input);

        qcu::device::tensorop::wilson_dslash_sun_mrhs_backward_ghost_pack
        <Float_, BlockShape, WarpShape>
            <<<grid_size, block_size, 0, bwd_stream>>> (
                static_cast<Float_*>(d_bwd_pack_buf), static_cast<Float_*>(dslash_param.fermion_in_MRHS),
                static_cast<Float_*>(dslash_param.gauge), ghost_dim, latt_desc,
                multiprocess,
                dslash_param.parity, dslash_param.dagger_flag,
                dslash_param.n_color, dslash_param.m_input);
    }
    // CHECK_CUDA(cudaDeviceSynchronize());
    if (!config::cuda_aware_mpi_supported()) {
        void* h_fwd_pack_buf = dslash_param.fermion_ghost->get_host_pack_buf_at(ghost_dim, FWD);
        void* h_bwd_pack_buf = dslash_param.fermion_ghost->get_host_pack_buf_at(ghost_dim, BWD);

        int byte_size = dslash_param.fermion_ghost->ghost_len[ghost_dim] * type_size;
        CHECK_CUDA(cudaMemcpyAsync(h_fwd_pack_buf, d_fwd_pack_buf, byte_size, cudaMemcpyDeviceToHost, fwd_stream));
        CHECK_CUDA(cudaMemcpyAsync(h_bwd_pack_buf, d_bwd_pack_buf, byte_size, cudaMemcpyDeviceToHost, bwd_stream));
    }
#else
    errorQcu("TensorOp not supported\n");
#endif
}

void WilsonDslash::pre_apply(const std::shared_ptr<DslashParam> dslash_param) {
    if (config::get_mpi_separated_mask() > 0) {
        size_t type_size = 0;
        switch (dslash_param->dslash_precision) {
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

        // Launch Kernel and MemcpyAsync (if donnot support cuda-aware MPI)
        for (int mu = 0; mu < Nd; ++mu) {
            if (dslash_param->proc_desc->at(mu) > 1) {
                switch (dslash_param->dslash_precision) {
                    case QcuPrecision::kPrecisionHalf:
                    {   apply_sun_mrhs_dslash_ghost_pack<half>(*dslash_param, mu);    }
                    break;
                    case QcuPrecision::kPrecisionSingle:
                    {   apply_sun_mrhs_dslash_ghost_pack<float>(*dslash_param, mu);   }
                    break;
                    case QcuPrecision::kPrecisionDouble:
                    {   apply_sun_mrhs_dslash_ghost_pack<double>(*dslash_param, mu); }
                    break;
                    default:
                    {   errorQcu("Not implemented yet\n"); }
                    break;
                }
            }
        }

        // Barrier
        CHECK_CUDA(cudaDeviceSynchronize());

        // Send
        QcuProcDesc proc_desc = *(dslash_param->proc_desc);

        FourDimDesc mpi_desc {proc_desc.data[X_DIM], proc_desc.data[Y_DIM],proc_desc.data[Z_DIM], proc_desc.data[T_DIM]};

        for (int mu = 0; mu < Nd; ++mu) {

            qcu::FourDimCoordinate mpi_coord_forward = config::get_mpi_coord();
            mpi_coord_forward.data[mu] = (mpi_coord_forward.data[mu] + 1) % proc_desc.data[mu];

            qcu::FourDimCoordinate mpi_coord_backward = config::get_mpi_coord();
            mpi_coord_backward.data[mu] = (mpi_coord_backward.data[mu] - 1 + proc_desc.data[mu]) % proc_desc.data[mu];

            int byte_size = dslash_param->fermion_ghost->ghost_len[mu] * type_size;

            void* fwd_sendbuf = nullptr;
            void* bwd_sendbuf = nullptr;
            void* fwd_recvbuf = nullptr;
            void* bwd_recvbuf = nullptr;
            if (config::cuda_aware_mpi_supported()) {
                fwd_sendbuf = dslash_param->fermion_ghost->get_pack_buf_at(mu, FWD);
                bwd_sendbuf = dslash_param->fermion_ghost->get_pack_buf_at(mu, BWD);

                fwd_recvbuf = dslash_param->fermion_ghost->get_unpack_buf_at(mu, FWD);
                bwd_recvbuf = dslash_param->fermion_ghost->get_unpack_buf_at(mu, BWD);
            }
            else {
                fwd_sendbuf = dslash_param->fermion_ghost->get_host_pack_buf_at(mu, FWD);
                bwd_sendbuf = dslash_param->fermion_ghost->get_host_pack_buf_at(mu, BWD);

                fwd_recvbuf = dslash_param->fermion_ghost->get_host_unpack_buf_at(mu, FWD);
                bwd_recvbuf = dslash_param->fermion_ghost->get_host_unpack_buf_at(mu, BWD);
            }

            if (dslash_param->proc_desc->at(mu) > 1) {
                CHECK_MPI(
                    MPI_Irecv(
                        fwd_recvbuf,
                        byte_size, MPI_BYTE,
                        mpi_coord_forward.getReversedIdx1D(mpi_desc),
                        FWD,
                        MPI_COMM_WORLD,
                        &config::get_mpi_request_unpack(mu, FWD))
                );
                CHECK_MPI(
                    MPI_Irecv(
                        bwd_recvbuf,
                        byte_size,
                        MPI_BYTE,
                        mpi_coord_backward.getReversedIdx1D(mpi_desc),
                        BWD,
                        MPI_COMM_WORLD,
                        &config::get_mpi_request_unpack(mu, BWD))
                );
                CHECK_MPI(
                    MPI_Isend(
                        bwd_sendbuf,
                        byte_size,
                        MPI_BYTE,
                        mpi_coord_backward.getReversedIdx1D(mpi_desc),
                        FWD,
                        MPI_COMM_WORLD,
                        &config::get_mpi_request_pack(mu, BWD)
                    )
                );
                CHECK_MPI(
                    MPI_Isend(
                        fwd_sendbuf,
                        byte_size,
                        MPI_BYTE,
                        mpi_coord_forward.getReversedIdx1D(mpi_desc),
                        BWD,
                        MPI_COMM_WORLD,
                        &config::get_mpi_request_pack(mu, FWD)
                    )
                );
            }
            else {
                config::get_mpi_request_pack(mu, FWD) = MPI_REQUEST_NULL;
                config::get_mpi_request_pack(mu, BWD) = MPI_REQUEST_NULL;
                config::get_mpi_request_unpack(mu, FWD) = MPI_REQUEST_NULL;
                config::get_mpi_request_unpack(mu, BWD) = MPI_REQUEST_NULL;
            }
        }
    }

}


}
