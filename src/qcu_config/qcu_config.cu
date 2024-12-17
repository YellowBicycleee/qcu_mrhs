// Created by YellowBicycleee on 24-10-21.
// Create this file to record variables used frequently in the Qcu class,
//        to avoid repeated computation and improve efficiency.

#include <check_error/check_mpi.h>
#include <lattice_desc.h>
#include <mpi.h>

#include <cstdint>

#include "check_error/check_cuda.cuh"
#include "desc/qcu_desc.h"
#include "qcu_config/qcu_config.h"
#include "qcu_helper.h"
namespace qcu::config {

constexpr int kQcuCudaStreamNum = 9; // 4 dim * 2 dir (0 ~ 7) + 1 central stream (8)
static cudaStream_t stream_pack[kQcuCudaStreamNum] = {nullptr};
static qcu::QcuLattDesc lattice_desc; // record the lattice size in single process (rather than total lattice)
static qcu::QcuProcDesc process_desc;
static qcu::FourDimCoordinate mpi_coord {-1, -1, -1, -1};
static int mpi_rank = 0;

int32_t lattice_volume() {
    return lattice_desc.lattice_volume();
}

int32_t whole_lattice_volume() {
    return lattice_desc.lattice_volume() * process_desc.process_volume();
}

bool set_config(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt){
    int mpi_comm_size;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size));
    assert(mpi_comm_size == Gx * Gy * Gz * Gt);

    lattice_desc = qcu::QcuLattDesc(Lx, Ly, Lz, Lt);
    process_desc = qcu::QcuProcDesc(Gx, Gy, Gz, Gt);

    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    mpi_coord.data[X_DIM] = mpi_rank % Gx;
    mpi_coord.data[Y_DIM] = (mpi_rank / Gx) % Gy;
    mpi_coord.data[Z_DIM] = (mpi_rank / Gx / Gy) % Gz;
    mpi_coord.data[T_DIM] = mpi_rank / Gx / Gy / Gz;

    return true;
}

qcu::QcuLattDesc* get_lattice_desc_ptr() {
    return &lattice_desc;
}

qcu::QcuProcDesc* get_process_desc_ptr() {
    return &process_desc;
}

qcu::FourDimCoordinate get_mpi_coord() {
    return mpi_coord;
}

qcu::FourDimDesc get_mpi_desc() {
    return qcu::FourDimDesc{
        process_desc.data[X_DIM], process_desc.data[Y_DIM],
        process_desc.data[Z_DIM], process_desc.data[T_DIM]};
}
qcu::FourDimDesc get_latt_desc() {
    return qcu::FourDimDesc{
        lattice_desc.data[X_DIM], lattice_desc.data[Y_DIM],
        lattice_desc.data[Z_DIM], lattice_desc.data[T_DIM]};
}

int get_mpi_rank() {
    return mpi_rank;
}

// cuda stream functions
constexpr int get_qcu_stream_num() noexcept {
    return kQcuCudaStreamNum;
}

constexpr cudaStream_t* get_qcu_stream_ptr() noexcept {
    return stream_pack;
}

cudaStream_t get_qcu_default_stream() noexcept {
    return stream_pack[0];
}

void init_streams() {
#pragma unroll
    for (int i = 0; i < kQcuCudaStreamNum; ++i) {
        CHECK_CUDA(cudaStreamCreate(&stream_pack[i]));
    }
}

void destroy_streams() {
#pragma unroll
    for (int i = 0; i < kQcuCudaStreamNum; ++i) {
        CHECK_CUDA(cudaStreamDestroy(stream_pack[i]));
    }
}

}
