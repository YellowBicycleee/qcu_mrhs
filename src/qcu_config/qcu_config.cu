// Created by YellowBicycleee on 24-10-21.
// Create this file to record variables used frequently in the Qcu class,
//        to avoid repeated computation and improve efficiency.

#include <check_error/check_mpi.h>
#include <lattice_desc.h>
#include <mpi.h>

#include <memory>

#include "check_error/check_cuda.cuh"
#include "desc/qcu_desc.h"
#include "qcu_config/qcu_config.h"

namespace qcu::config {

constexpr int kQcuCudaStreamNum = 9;  // 4 dim * 2 dir (0 ~ 7) + 1 central stream (8)
static std::vector<cudaStream_t> stream_pack;
static qcu::QcuLattDesc lattice_desc; // record the lattice size in single process (rather than total lattice)
static qcu::QcuProcDesc process_desc;

static std::vector<MPI_Request> mpi_request_pack = std::vector<MPI_Request>(8);
static std::vector<MPI_Request> mpi_request_unpack = std::vector<MPI_Request>(8);

MPI_Request& get_mpi_request_pack(int dim, int dir) {
    return mpi_request_pack[2 * dim + dir];
}
MPI_Request& get_mpi_request_unpack(int dim, int dir) {
    return mpi_request_unpack[2 * dim + dir];
}

// static qcu::FourDimCoordinate mpi_coord {-1, -1, -1, -1};

// assume: single thread setting
class QcuConfig {
public:
    QcuConfig(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt)
        : latt_vol_total(Lx * Ly * Lz * Lt * Gx * Gy * Gz * Gt)
        , latt_vol_local(Lx * Ly * Lz * Lt)
        , mpi_comm_size(Gx * Gy * Gz * Gt)
        , latt_desc(Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt)
        , latt_desc_local(Lx, Ly, Lz, Lt)
        , mpi_desc(Gx, Gy, Gz, Gt)
    {
        if (Gx > 1) { mpi_separated_mask |= (1 << X_DIM); }
        if (Gy > 1) { mpi_separated_mask |= (1 << Y_DIM); }
        if (Gz > 1) { mpi_separated_mask |= (1 << Z_DIM); }
        if (Gt > 1) { mpi_separated_mask |= (1 << T_DIM); }

        // setting mpi parameters
        CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_comm_rank));
        // mpi use reverse coord
        mpi_coord = qcu::FourDimCoordinate{
            mpi_comm_rank / (Gy * Gz * Gt),
            (mpi_comm_rank / Gz / Gt) % Gy,
            (mpi_comm_rank / Gt) % Gz,
            mpi_comm_rank % Gt};
        // mpi_coord = qcu::FourDimCoordinate{
        //     mpi_comm_rank % Gx,
        //     (mpi_comm_rank / Gx) % Gy,
        //     (mpi_comm_rank / Gx / Gy) % Gz,
        //     mpi_comm_rank / Gx / Gy / Gz};
    }

    [[nodiscard]] int get_lattice_vol_total () const { return latt_vol_total;   }
    [[nodiscard]] int get_lattice_vol_local () const { return latt_vol_local;   }
    [[nodiscard]] int get_mpi_comm_size () const { return mpi_comm_size; }
    [[nodiscard]] int get_mpi_comm_rank () const { return mpi_comm_rank; }
    [[nodiscard]] unsigned int get_mpi_separated_mask () const { return mpi_separated_mask; }
    [[nodiscard]] qcu::QcuLattDesc get_latt_desc () const { return latt_desc; }
    [[nodiscard]] qcu::QcuLattDesc get_latt_desc_local () const { return latt_desc; }
    [[nodiscard]] qcu::QcuProcDesc get_mpi_desc () const { return mpi_desc; }
    [[nodiscard]] qcu::FourDimCoordinate get_mpi_coord() const { return mpi_coord; }
    [[nodiscard]] bool process_last_t () const { return mpi_coord.data[T_DIM] == mpi_desc.data[T_DIM] - 1; }
private:
    int latt_vol_total;
    int latt_vol_local;
    int mpi_comm_size;
    int mpi_comm_rank;
    unsigned int mpi_separated_mask = 0;
    qcu::QcuLattDesc latt_desc;
    qcu::QcuLattDesc latt_desc_local;
    qcu::QcuProcDesc mpi_desc;
    qcu::FourDimCoordinate mpi_coord{-1, -1, -1, -1};
};

std::shared_ptr<QcuConfig> qcu_configuration(nullptr);

bool is_last_process_t() {
    if (qcu_configuration) { return qcu_configuration->process_last_t(); }
    else { errorQcu("Get parameters before configured\n"); }
}

unsigned int get_mpi_separated_mask () {
    if (qcu_configuration) { return qcu_configuration->get_mpi_separated_mask(); }
    else { errorQcu("Get parameters before configured\n"); }
}

bool set_config(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt){
    int mpi_comm_size;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size));
    assert(mpi_comm_size == Gx * Gy * Gz * Gt);

    lattice_desc = qcu::QcuLattDesc(Lx, Ly, Lz, Lt);
    process_desc = qcu::QcuProcDesc(Gx, Gy, Gz, Gt);

    qcu_configuration = std::make_shared<QcuConfig>(Lx, Ly, Lz, Lt, Gx, Gy, Gz, Gt);
    return true;
}

int lattice_volume_local() {
    if (qcu_configuration) { return qcu_configuration->get_lattice_vol_local(); }
    else { errorQcu("Get parameters before configured\n"); }
}

int lattice_volume_total() {
    if (qcu_configuration) { return qcu_configuration->get_lattice_vol_total(); }
    else { errorQcu("Get parameters before configured\n"); }
}

qcu::QcuLattDesc* get_lattice_desc_ptr() {
    return &lattice_desc;
}

qcu::QcuProcDesc* get_process_desc_ptr() {
    return &process_desc;
}

qcu::FourDimCoordinate get_mpi_coord() {
    assert(qcu_configuration != nullptr);
    return qcu_configuration->get_mpi_coord();
}

// qcu::FourDimDesc get_mpi_desc() {
//     return qcu::FourDimDesc{
//         process_desc.data[X_DIM], process_desc.data[Y_DIM],
//         process_desc.data[Z_DIM], process_desc.data[T_DIM]};
// }

std::vector<int> get_mpi_desc() {
    qcu::QcuProcDesc process_desc = qcu_configuration->get_mpi_desc();
    return {
        process_desc.X(), process_desc.Y(), process_desc.Z(), process_desc.T()};
}
std::vector<int> get_latt_desc() {
    qcu::QcuLattDesc latt_desc = qcu_configuration->get_latt_desc();
    return {latt_desc.X(), latt_desc.Y(), latt_desc.Z(), latt_desc.T()};
}

int get_mpi_rank() {
    return qcu_configuration->get_mpi_comm_rank();
}

// cuda stream functions
constexpr int get_qcu_stream_num() noexcept {
    return kQcuCudaStreamNum;
}

std::vector<cudaStream_t>& get_qcu_streams() noexcept {
    return stream_pack;
}

cudaStream_t get_qcu_default_stream() noexcept {
    return stream_pack[kQcuCudaStreamNum - 1];
}

void init_streams() {
#pragma unroll
    for (int i = 0; i < kQcuCudaStreamNum; ++i) {
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        stream_pack.push_back(stream);
    }
}

void destroy_streams() {
#pragma unroll
    for (int i = 0; i < kQcuCudaStreamNum; ++i) {
        CHECK_CUDA(cudaStreamDestroy(stream_pack[i]));
    }
    stream_pack.clear();
}

}
