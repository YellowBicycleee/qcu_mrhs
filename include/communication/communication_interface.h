//
// Created by wjc on 24-10-28.
//

#pragma once
#include "communication/communicator.h"
#include <string>
namespace qcu {
namespace communication {

QcuCommStatus qcu_mpi_init ();
QcuCommStatus qcu_nccl_init();

QcuCommModel qcu_set_comm_model (QcuCommModel preferred_comm_model = QcuCommModel::kQcuCommNccl);
QcuCommStatus send();
QcuCommStatus isend();
QcuCommStatus recv();
QcuCommStatus irecv();
QcuCommStatus reduce_sum();
QcuCommStatus barrier();

std::string get_qcu_comm_status_string(QcuCommStatus status);
}
}