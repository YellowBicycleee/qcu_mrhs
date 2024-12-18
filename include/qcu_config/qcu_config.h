#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "desc/qcu_desc.h"
#include "lattice_desc.h"

namespace qcu {
namespace config {

bool set_config(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt);
int lattice_volume_local();
int lattice_volume_total();
qcu::QcuLattDesc* get_lattice_desc_ptr();
qcu::QcuProcDesc* get_process_desc_ptr();
qcu::FourDimCoordinate get_mpi_coord();
qcu::FourDimDesc get_mpi_desc();
qcu::FourDimDesc get_latt_desc();
int get_mpi_rank();

constexpr int get_qcu_stream_num() noexcept;
std::vector<cudaStream_t>& get_qcu_streams() noexcept;
cudaStream_t get_qcu_default_stream() noexcept;
void init_streams();
void destroy_streams();
}
}