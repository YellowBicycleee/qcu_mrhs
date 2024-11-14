//
// Created by Wang Jiancheng on 24-10-21.
//
// Create this file to record variables used frequently in the Qcu class,
//        to avoid repeated computation and improve efficiency.

#include <cstdint>

#include "desc/qcu_desc.h"
#include "qcu_config/qcu_config.h"
#include "qcu_helper.h"
#include "check_error/check_cuda.cuh"
namespace qcu::config {

constexpr int kQcuCudaStreamNum = 9;

static cudaStream_t stream_pack[kQcuCudaStreamNum] = {nullptr};
static qcu::QcuLattDesc lattice_desc; // record the lattice size in single process (rather than total lattice)
static qcu::QcuProcDesc process_desc;

int32_t lattice_volume() {
    return lattice_desc.lattice_volume();
}
int32_t whole_lattice_volume() {
    return lattice_desc.lattice_volume() * process_desc.process_volume();
}
bool set_config(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt){
    lattice_desc = qcu::QcuLattDesc(Lx, Ly, Lz, Lt);
    process_desc = qcu::QcuProcDesc(Gx, Gy, Gz, Gt);
    return true;
}

qcu::QcuLattDesc* get_lattice_desc_ptr() {
    return &lattice_desc;
}
qcu::QcuProcDesc* get_process_desc_ptr() {
    return &process_desc;
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
