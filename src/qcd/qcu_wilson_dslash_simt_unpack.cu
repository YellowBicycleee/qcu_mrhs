/**
 * @file qcu_wilson_dslash_simt_unpack.cu
 *
 * @brief Kernel for unpacking the ghost for the Wilson Dslash operator
 *
 */
#include <mpi.h>
#include "qcd/qcu_dslash_wilson.h"

#include "kernel/sun_mrhs_wilson_dslash_unpack_simt.cuh"
#include "qcu_config/qcu_config.h"
#include "check_error/check_cuda.cuh"
#include "check_error/check_mpi.h"

namespace qcu::simt {
// unpack走8流
template <typename Float_>
inline void apply_sun_mrhs_dslash_ghost_unpack ( DslashParam& dslash_param, int ghost_dim) {
    using BlockShape = gemm::GemmShape<8, 8, 8>;

    const qcu::QcuLattDesc& latt_desc = *(dslash_param.latt_desc);

    int half_vol = config::lattice_volume_local() / 2;
    int num_threads = half_vol / latt_desc.at(ghost_dim);

    int blk_x = BlockShape::kM;
    int blk_y = BlockShape::kN;

    cudaStream_t fwd_stream = dslash_param.streams[8];
    cudaStream_t bwd_stream = dslash_param.streams[8];

    void* fwd_unpack_buf = dslash_param.fermion_ghost->get_unpack_buf_at(ghost_dim, FWD);
    void* bwd_unpack_buf = dslash_param.fermion_ghost->get_unpack_buf_at(ghost_dim, BWD);

    dim3 grid_size(div_ceil(dslash_param.n_color, blk_x), div_ceil(dslash_param.m_input, blk_y), std::min(num_threads, 65535));
    dim3 block_size(blk_x, blk_y, 1);

    // printf("SIMT dslash pack Beginning\n");
    qcu::device::wilson_dslash_sun_mrhs_forward_ghost_unpack<Float_, BlockShape>
        <<<grid_size, block_size, 0, fwd_stream>>> (
            static_cast<Float_*>(dslash_param.fermion_out_MRHS), static_cast<Float_*>(fwd_unpack_buf),
            static_cast<Float_*>(dslash_param.gauge), latt_desc, ghost_dim, dslash_param.parity,
            dslash_param.dagger_flag, dslash_param.n_color, dslash_param.m_input);
    qcu::device::wilson_dslash_sun_mrhs_backward_ghost_unpack<Float_, BlockShape>
        <<<grid_size, block_size, 0, bwd_stream>>> (
            static_cast<Float_*>(dslash_param.fermion_out_MRHS), static_cast<Float_*>(bwd_unpack_buf),
            latt_desc, ghost_dim, dslash_param.parity, dslash_param.dagger_flag,
            dslash_param.n_color, dslash_param.m_input);
    CHECK_CUDA(cudaDeviceSynchronize());
    // printf("SIMT dslash Ending, config = grid(%d, %d, %d), block(%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
}

void WilsonDslash::apply_ghost_unpack(DslashParam& dslash_param, int ghost_dim) {

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

    FourDimDesc mpi_desc {proc_desc.data[X_DIM], proc_desc.data[Y_DIM], proc_desc.data[Z_DIM], proc_desc.data[T_DIM]};

    const int byte_size = dslash_param.fermion_ghost->ghost_len[ghost_dim] * type_size;

    void* device_unpack_buf_fwd = dslash_param.fermion_ghost->get_unpack_buf_at(ghost_dim, FWD);
    void* device_unpack_buf_bwd = dslash_param.fermion_ghost->get_unpack_buf_at(ghost_dim, BWD);
    void* host_unpack_buf_fwd = dslash_param.fermion_ghost->get_host_unpack_buf_at(ghost_dim, FWD);
    void* host_unpack_buf_bwd = dslash_param.fermion_ghost->get_host_unpack_buf_at(ghost_dim, BWD);

    CHECK_MPI(
        MPI_Irecv(host_unpack_buf_fwd, byte_size, MPI_BYTE, mpi_coord_forward.getReversedIdx1D(mpi_desc),
            FWD, MPI_COMM_WORLD, &config::get_mpi_request_unpack(ghost_dim, FWD))
    );

    CHECK_MPI(
        MPI_Irecv(host_unpack_buf_bwd, byte_size, MPI_BYTE, mpi_coord_backward.getReversedIdx1D(mpi_desc),
            BWD, MPI_COMM_WORLD, &config::get_mpi_request_unpack(ghost_dim, BWD))
    );

    CHECK_MPI(MPI_Wait(&config::get_mpi_request_pack(ghost_dim, FWD), MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&config::get_mpi_request_pack(ghost_dim, BWD), MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&config::get_mpi_request_unpack(ghost_dim, FWD), MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Wait(&config::get_mpi_request_unpack(ghost_dim, BWD), MPI_STATUS_IGNORE));
    CHECK_CUDA(cudaMemcpy(device_unpack_buf_fwd, host_unpack_buf_fwd, byte_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_unpack_buf_bwd, host_unpack_buf_bwd, byte_size, cudaMemcpyHostToDevice));

    switch (dslash_param.dslash_precision) {
        case QcuPrecision::kPrecisionHalf:
            {   apply_sun_mrhs_dslash_ghost_unpack<half>(dslash_param, ghost_dim);  }
            break;
        case QcuPrecision::kPrecisionSingle:
            {   apply_sun_mrhs_dslash_ghost_unpack<float>(dslash_param, ghost_dim); }
        break;
        case QcuPrecision::kPrecisionDouble:
            {   apply_sun_mrhs_dslash_ghost_unpack<double>(dslash_param, ghost_dim); }
            break;
        default:
            {   errorQcu("Precision must be one of {half, single, double}\n"); }
            break;
    }
    CHECK_CUDA(cudaStreamSynchronize(dslash_param.streams[8]));
}
}