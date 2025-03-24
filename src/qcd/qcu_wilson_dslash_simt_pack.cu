//
// Created by wangj on 2024/12/18.
//
#include <mpi.h>

#include "check_error/check_cuda.cuh"
#include "check_error/check_mpi.h"
#include "kernel/sun_mrhs_wilson_dslash_simt_pack.cuh"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_config/qcu_config.h"

namespace qcu::simt {
// pack 使用独立的8个流
template <typename Float_> // Launch Kernel and MemcpyAsync (if donnot support cuda-aware MPI)
inline void apply_sun_mrhs_dslash_ghost_pack (DslashParam& dslash_param, int ghost_dim) {

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

    using BlockShape = gemm::GemmShape<8, 8, 8>;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    int half_vol = config::lattice_volume_local() / 2;
    int num_threads = half_vol / latt_desc.at(ghost_dim);

    int blk_x = BlockShape::kM;
    int blk_y = BlockShape::kN;

    void* d_fwd_pack_buf = dslash_param.fermion_ghost->get_pack_buf_at(ghost_dim, FWD);
    void* d_bwd_pack_buf = dslash_param.fermion_ghost->get_pack_buf_at(ghost_dim, BWD);

    dim3 grid_size(div_ceil(dslash_param.n_color, blk_x), div_ceil(dslash_param.m_input, blk_y), std::min(num_threads, 65535));
    dim3 block_size(blk_x, blk_y, 1);

    cudaStream_t fwd_stream = dslash_param.streams[ghost_dim * 2 + FWD];
    cudaStream_t bwd_stream = dslash_param.streams[ghost_dim * 2 + BWD];

    qcu::device::wilson_dslash_sun_mrhs_forward_ghost_pack<Float_, BlockShape>
        <<<grid_size, block_size, 0, fwd_stream>>> (
            static_cast<Float_*>(d_fwd_pack_buf), static_cast<Float_*>(dslash_param.fermion_in_MRHS),
            static_cast<Float_*>(dslash_param.gauge), latt_desc, ghost_dim,
            dslash_param.parity, dslash_param.dagger_flag,
            dslash_param.n_color, dslash_param.m_input);

    qcu::device::wilson_dslash_sun_mrhs_backward_ghost_pack<Float_, BlockShape>
        <<<grid_size, block_size, 0, bwd_stream>>> (
            static_cast<Float_*>(d_bwd_pack_buf), static_cast<Float_*>(dslash_param.fermion_in_MRHS),
            latt_desc, ghost_dim, dslash_param.parity, dslash_param.dagger_flag,
            dslash_param.n_color, dslash_param.m_input);
    // CHECK_CUDA(cudaDeviceSynchronize());

    if (!config::cuda_aware_mpi_supported()) {
        void* h_fwd_pack_buf = dslash_param.fermion_ghost->get_host_pack_buf_at(ghost_dim, FWD);
        void* h_bwd_pack_buf = dslash_param.fermion_ghost->get_host_pack_buf_at(ghost_dim, BWD);

        int byte_size = dslash_param.fermion_ghost->ghost_len[ghost_dim] * type_size;
        CHECK_CUDA(cudaMemcpyAsync(h_fwd_pack_buf, d_fwd_pack_buf, byte_size, cudaMemcpyDeviceToHost, fwd_stream));
        CHECK_CUDA(cudaMemcpyAsync(h_bwd_pack_buf, d_bwd_pack_buf, byte_size, cudaMemcpyDeviceToHost, bwd_stream));
    }
}


void WilsonDslash::pre_apply(const std::shared_ptr<DslashParam> dslash_param) {

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


    // std::vector<void*> send_buf(Nd * 2); // 只支持4维
    //
    // // 确定发送缓冲区为设备端还是主机端
    // for (int mu = 0; mu < Nd; ++mu) {
    //     int byte_size = dslash_param->fermion_ghost->ghost_len[mu] * type_size;
    //
    //     if (qcu::config::cuda_aware_mpi_supported()) {
    //         send_buf[mu * 2 + FWD] = dslash_param->fermion_ghost->get_pack_buf_at(mu, FWD);
    //         send_buf[mu * 2 + BWD] = dslash_param->fermion_ghost->get_pack_buf_at(mu, BWD);
    //     }
    //     else {
    //         send_buf[mu * 2 + FWD] = dslash_param->fermion_ghost->get_host_pack_buf_at(mu, FWD);
    //         send_buf[mu * 2 + BWD] = dslash_param->fermion_ghost->get_host_pack_buf_at(mu, BWD);
    //     }
    // }

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
        if (config::cuda_aware_mpi_supported()) {
            fwd_sendbuf = dslash_param->fermion_ghost->get_pack_buf_at(mu, FWD);
            bwd_sendbuf = dslash_param->fermion_ghost->get_pack_buf_at(mu, BWD);
        }
        else {
            fwd_sendbuf = dslash_param->fermion_ghost->get_host_pack_buf_at(mu, FWD);
            bwd_sendbuf = dslash_param->fermion_ghost->get_host_pack_buf_at(mu, BWD);
        }

        if (dslash_param->proc_desc->at(mu) > 1) {
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
                    mpi_coord_backward.getReversedIdx1D(mpi_desc),
                    BWD,
                    MPI_COMM_WORLD,
                    &config::get_mpi_request_pack(mu, FWD)
                )
            );
        }
    }

}

}
