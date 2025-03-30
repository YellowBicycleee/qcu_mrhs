#include "qcd/qcu_dslash_staggered.h"
#include "kernel/sun_mrhs_staggered_dslash_simt_pack.cuh"
#include "qcu_config/qcu_config.h"
#include "qcu_utils.h"
#include <mpi.h>
#include "check_error/check_cuda.cuh"
#include "check_error/check_mpi.h"
#include "kernel/gemm/qcu_gemm_configure.cuh"
namespace qcu::simt {
// pack 使用独立的8个流
template <typename Float_>
inline void apply_sun_mrhs_dslash_ghost_pack (DslashParam& dslash_param, int ghost_dim) {

}

void StaggeredDslash::pre_apply(const std::shared_ptr<DslashParam> dslash_param) {
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
                        mpi_coord_backward.getReversedIdx1D(mpi_desc),
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