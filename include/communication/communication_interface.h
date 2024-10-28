//
// Created by wjc on 24-10-28.
//

#pragma once
#include "communication/communicator.h"
namespace qcu {
namespace communication {


QcuCommModel qcu_set_comm_model (QcuCommModel preferred_comm_model = QcuCommModel::kQcuCommNccl);
QcuCommStatus send();
QcuCommStatus isend();
QcuCommStatus recv();
QcuCommStatus irecv();
QcuCommStatus reduce_sum();
QcuCommStatus barrier();


}
}