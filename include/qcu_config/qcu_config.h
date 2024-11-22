//
// Created by wjc on 24-10-21.
//
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "desc/qcu_desc.h"

namespace qcu {
namespace config {

int32_t lattice_volume();
int32_t whole_lattice_volume();
bool set_config(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt);
qcu::QcuLattDesc* get_lattice_desc_ptr();
qcu::QcuProcDesc* get_process_desc_ptr();

constexpr int get_qcu_stream_num() noexcept;
constexpr cudaStream_t* get_qcu_stream_ptr() noexcept;
cudaStream_t get_qcu_default_stream() noexcept;
void init_streams();
void destroy_streams();
}
}