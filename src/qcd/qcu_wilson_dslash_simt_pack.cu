//
// Created by wangj on 2024/12/18.
//
#include <mpi.h>

#include "check_error/check_cuda.cuh"
#include "kernel/sun_mrhs_wilson_dslash_pack_simt.cuh"
#include "qcd/qcu_dslash_wilson.h"
#include "qcu_config/qcu_config.h"

namespace qcu::simt {

template <typename Float_>
inline void apply_sun_mrhs_dslash_forward_ghost_pack ( DslashParam& dslash_param, int ghost_dim) {
    using BlockShape = gemm::GemmShape<8, 8, 8>;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    int half_vol = config::lattice_volume_local() / 2;
    int num_threads = half_vol / latt_desc.at(ghost_dim);

    int blk_x = BlockShape::kM;
    int blk_y = BlockShape::kN;

    void* pack_buf = dslash_param.fermion_ghost->get_pack_buf_at(2 * ghost_dim + FWD);
    dim3 grid_size(div_ceil(dslash_param.n_color, blk_x), div_ceil(dslash_param.m_input, blk_y), std::min(num_threads, 65535));
    dim3 block_size(blk_x, blk_y, 1);

    printf("SIMT dslash pack Beginning\n");
    qcu::device::wilson_dslash_sun_mrhs_forward_ghost_pack<Float_, BlockShape> <<<grid_size, block_size, 0, dslash_param.stream1>>>
        (static_cast<Float_*>(pack_buf), static_cast<Float_*>(dslash_param.fermion_in_MRHS),
        static_cast<Float_*>(dslash_param.gauge), latt_desc, ghost_dim, dslash_param.parity, dslash_param.dagger_flag,
        dslash_param.n_color, dslash_param.m_input);
    CHECK_CUDA(cudaDeviceSynchronize());
    // printf("SIMT dslash Ending, config = grid(%d, %d, %d), block(%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
}

template <typename Float_>
inline void apply_sun_mrhs_dslash_backward_ghost_pack ( DslashParam& dslash_param, int ghost_dim) {
    using BlockShape = gemm::GemmShape<8, 8, 8>;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    int half_vol = config::lattice_volume_local() / 2;
    int num_threads = half_vol / latt_desc.at(ghost_dim);

    int blk_x = BlockShape::kM;
    int blk_y = BlockShape::kN;

    void* pack_buf = dslash_param.fermion_ghost->get_pack_buf_at(2 * ghost_dim + BWD);
    dim3 grid_size(div_ceil(dslash_param.n_color, blk_x), div_ceil(dslash_param.m_input, blk_y), std::min(num_threads, 65535));
    dim3 block_size(blk_x, blk_y, 1);

    printf("SIMT dslash pack Beginning\n");
    qcu::device::wilson_dslash_sun_mrhs_backward_ghost_pack<Float_, BlockShape> <<<grid_size, block_size, 0, dslash_param.stream1>>>
        (static_cast<Float_*>(pack_buf), static_cast<Float_*>(dslash_param.fermion_in_MRHS),
        latt_desc, ghost_dim, dslash_param.parity, dslash_param.dagger_flag, dslash_param.n_color, dslash_param.m_input);
    CHECK_CUDA(cudaDeviceSynchronize());
    // printf("SIMT dslash Ending, config = grid(%d, %d, %d), block(%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
}

void WilsonDslash::apply_ghost_pack(DslashParam& dslash_param, int ghost_dim) {

    switch (dslash_param.dslash_precision) {
        case QcuPrecision::kPrecisionHalf: {
            apply_sun_mrhs_dslash_forward_ghost_pack<half>(dslash_param, ghost_dim);
            apply_sun_mrhs_dslash_backward_ghost_pack<half>(dslash_param, ghost_dim);
        }
        break;
        case QcuPrecision::kPrecisionSingle: {
            apply_sun_mrhs_dslash_forward_ghost_pack<float>(dslash_param, ghost_dim);
            apply_sun_mrhs_dslash_backward_ghost_pack<float>(dslash_param, ghost_dim);
        }
        break;
        case QcuPrecision::kPrecisionDouble: {
            apply_sun_mrhs_dslash_forward_ghost_pack<double>(dslash_param, ghost_dim);            apply_sun_mrhs_dslash_backward_ghost_pack<half>(dslash_param, ghost_dim);
            apply_sun_mrhs_dslash_backward_ghost_pack<double>(dslash_param, ghost_dim);
        }
        break;
        default:
        {
            errorQcu("Not implemented yet\n");
            assert(0);
        }
        break;
    }
    CHECK_CUDA(cudaStreamSynchronize(dslash_param.stream1));

    QcuProcDesc proc_desc = *(dslash_param.proc_desc);
    qcu::FourDimCoordinate mpi_coord_forward = config::get_mpi_coord();
    mpi_coord_forward.data[ghost_dim] = (mpi_coord_forward.data[ghost_dim] + 1) % proc_desc.data[ghost_dim];

    qcu::FourDimCoordinate mpi_coord_backward = config::get_mpi_coord();
    mpi_coord_backward.data[ghost_dim] = (mpi_coord_backward.data[ghost_dim] - 1 + proc_desc.data[ghost_dim]) % proc_desc.data[ghost_dim];

    size_t type_size = 0;
    switch (dslash_param.dslash_precision) {
        case kPrecisionDouble:
            type_size = sizeof(double);
            break;
        case kPrecisionSingle:
            type_size = sizeof(float);
            break;
        case kPrecisionHalf:
            type_size = sizeof(half);
            break;
        default:
            type_size = 0;
            break;
    }

    FourDimDesc mpi_desc {
        proc_desc.data[X_DIM], proc_desc.data[Y_DIM],
        proc_desc.data[Z_DIM], proc_desc.data[T_DIM]
    };
    const int byte_size = dslash_param.fermion_ghost->ghost_len[ghost_dim] * type_size;

    void* device_pack_buf_fwd = dslash_param.fermion_ghost->get_pack_buf_at(2 * ghost_dim + FWD);
    void* host_pack_buf_fwd = dslash_param.fermion_ghost->get_host_pack_buf_at(2 * ghost_dim + BWD);
    cudaMemcpy(host_pack_buf_fwd, device_pack_buf_fwd, byte_size, cudaMemcpyDeviceToHost);
    MPI_Send(
        host_pack_buf_fwd,
        byte_size,
        MPI_BYTE,
        mpi_coord_forward.getIdx1D(mpi_desc),
        BWD,
        MPI_COMM_WORLD);

    void* device_pack_buf_bwd = dslash_param.fermion_ghost->get_pack_buf_at(2 * ghost_dim + BWD);
    void* host_pack_buf_bwd = dslash_param.fermion_ghost->get_host_pack_buf_at(2 * ghost_dim + BWD);
    cudaMemcpy(host_pack_buf_bwd, device_pack_buf_bwd, byte_size, cudaMemcpyDeviceToHost);
    MPI_Send(
        host_pack_buf_bwd,
        byte_size,
        MPI_BYTE,
        mpi_coord_backward.getIdx1D(mpi_desc),
        FWD,
        MPI_COMM_WORLD
    );

}

}