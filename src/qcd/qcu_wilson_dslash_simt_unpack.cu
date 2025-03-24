/**
 * @file qcu_wilson_dslash_simt_unpack.cu
 *
 * @brief Kernel for unpacking the ghost for the Wilson Dslash operator
 *
 */
#include <mpi.h>
#include "qcd/qcu_dslash_wilson.h"

#include "kernel/sun_mrhs_wilson_dslash_simt_unpack.cuh"
#include "qcu_config/qcu_config.h"
#include "check_error/check_cuda.cuh"
#include "check_error/check_mpi.h"

namespace qcu::simt {
// unpack走8流
template <typename Float_>
inline void apply_sun_mrhs_dslash_ghost_unpack ( DslashParam& dslash_param, int ghost_dim) {
    cudaStream_t fwd_stream = dslash_param.streams[8];
    cudaStream_t bwd_stream = dslash_param.streams[8];

    void* d_fwd_unpack_buf = dslash_param.fermion_ghost->get_unpack_buf_at(ghost_dim, FWD);
    void* d_bwd_unpack_buf = dslash_param.fermion_ghost->get_unpack_buf_at(ghost_dim, BWD);



    if (!config::cuda_aware_mpi_supported()) {
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
        void* h_fwd_unpack_buf = dslash_param.fermion_ghost->get_host_unpack_buf_at(ghost_dim, FWD);
        void* h_bwd_unpack_buf = dslash_param.fermion_ghost->get_host_unpack_buf_at(ghost_dim, BWD);

        int byte_size = dslash_param.fermion_ghost->ghost_len[ghost_dim] * type_size;
        CHECK_CUDA(cudaMemcpyAsync(d_fwd_unpack_buf, h_fwd_unpack_buf, byte_size, cudaMemcpyHostToDevice, fwd_stream));
        CHECK_CUDA(cudaMemcpyAsync(d_bwd_unpack_buf, h_bwd_unpack_buf, byte_size, cudaMemcpyHostToDevice, bwd_stream));
    }


    using BlockShape = gemm::GemmShape<8, 8, 8>;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    int half_vol = config::lattice_volume_local() / 2;
    int num_threads = half_vol / latt_desc.at(ghost_dim);

    int blk_x = BlockShape::kM;
    int blk_y = BlockShape::kN;


    dim3 grid_size(div_ceil(dslash_param.n_color, blk_x), div_ceil(dslash_param.m_input, blk_y), std::min(num_threads, 65535));
    dim3 block_size(blk_x, blk_y, 1);

    // printf("SIMT dslash pack Beginning\n");
    qcu::device::wilson_dslash_sun_mrhs_forward_ghost_unpack<Float_, BlockShape>
        <<<grid_size, block_size, 0, fwd_stream>>> (
            static_cast<Float_*>(dslash_param.fermion_out_MRHS), static_cast<Float_*>(d_fwd_unpack_buf),
            static_cast<Float_*>(dslash_param.gauge), latt_desc, ghost_dim, dslash_param.parity,
            dslash_param.dagger_flag, dslash_param.n_color, dslash_param.m_input);
    qcu::device::wilson_dslash_sun_mrhs_backward_ghost_unpack<Float_, BlockShape>
        <<<grid_size, block_size, 0, bwd_stream>>> (
            static_cast<Float_*>(dslash_param.fermion_out_MRHS), static_cast<Float_*>(d_bwd_unpack_buf),
            latt_desc, ghost_dim, dslash_param.parity, dslash_param.dagger_flag,
            dslash_param.n_color, dslash_param.m_input);
    // CHECK_CUDA(cudaDeviceSynchronize());
}

void WilsonDslash::post_apply(const std::shared_ptr<DslashParam> dslash_param) {
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

    // Wait MPI
    QcuProcDesc proc_desc = *(dslash_param->proc_desc);

    FourDimDesc mpi_desc {proc_desc.data[X_DIM], proc_desc.data[Y_DIM],proc_desc.data[Z_DIM], proc_desc.data[T_DIM]};

    for (int mu = 0; mu < Nd; ++mu) {

        qcu::FourDimCoordinate mpi_coord_forward = config::get_mpi_coord();
        mpi_coord_forward.data[mu] = (mpi_coord_forward.data[mu] + 1) % proc_desc.data[mu];

        qcu::FourDimCoordinate mpi_coord_backward = config::get_mpi_coord();
        mpi_coord_backward.data[mu] = (mpi_coord_backward.data[mu] - 1 + proc_desc.data[mu]) % proc_desc.data[mu];

        int byte_size = dslash_param->fermion_ghost->ghost_len[mu] * type_size;

        void* fwd_recvbuf = nullptr;
        void* bwd_recvbuf = nullptr;
        if (config::cuda_aware_mpi_supported()) {
            fwd_recvbuf = dslash_param->fermion_ghost->get_unpack_buf_at(mu, FWD);
            bwd_recvbuf = dslash_param->fermion_ghost->get_unpack_buf_at(mu, BWD);
        }
        else {
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
        }
        else {
            config::get_mpi_request_unpack(mu, BWD) = MPI_REQUEST_NULL;
            config::get_mpi_request_unpack(mu, FWD) = MPI_REQUEST_NULL;
        }
    }

    // Barrier
    std::vector<MPI_Request>& mpi_unpack_vec = config::get_mpi_request_unpack_vec();

    CHECK_MPI(
        MPI_Waitall(
            mpi_unpack_vec.size(),
            mpi_unpack_vec.data(),
            MPI_STATUSES_IGNORE
        )
    );

    for (int mu = 0; mu < Nd; ++mu) {
        if (dslash_param->proc_desc->at(mu) > 1) {
            switch (dslash_param->dslash_precision) {
                case QcuPrecision::kPrecisionHalf:
                {   apply_sun_mrhs_dslash_ghost_unpack<half>(*dslash_param, mu);    }
                break;
                case QcuPrecision::kPrecisionSingle:
                {   apply_sun_mrhs_dslash_ghost_unpack<float>(*dslash_param, mu);   }
                break;
                case QcuPrecision::kPrecisionDouble:
                {   apply_sun_mrhs_dslash_ghost_unpack<double>(*dslash_param, mu); }
                break;
                default:
                {   errorQcu("Not implemented yet\n"); }
                break;
            }
        }
    }

    // Barrier send
    std::vector<MPI_Request>& mpi_pack_vec = config::get_mpi_request_pack_vec();
    CHECK_MPI(
            MPI_Waitall(
            mpi_pack_vec.size(),
            mpi_pack_vec.data(),
            MPI_STATUSES_IGNORE
        )
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

}