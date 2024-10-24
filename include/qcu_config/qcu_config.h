//
// Created by wjc on 24-10-21.
//

#include <cstdint>
#include "desc/qcu_desc.h"
namespace qcu {
namespace config {

int32_t lattice_volume();
int32_t whole_lattice_volume();
bool set_config(int Lx, int Ly, int Lz, int Lt, int Gx, int Gy, int Gz, int Gt);
qcu::QcuLattDesc* get_lattice_desc_ptr();
qcu::QcuProcDesc* get_process_desc_ptr();

}
}