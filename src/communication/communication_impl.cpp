//
// Created by wjc on 24-10-28.
//

#include "check_error/check_mpi.h"
#include "communication/communicate_ghost.h"
#include "communication/communication_interface.h"
#include "qcu_public.h"

#include <memory>
#include <mpi.h>

namespace qcu {
namespace communication {
// for cuda p2p
static int mpi_rank = -1;
[[maybe_unused]] static int nccl_rank = -1;
[[maybe_unused]] static int p2p_tabel[Nd * DIRECTIONS] = {0}; // 2 * dim + dir

static QcuCommModel comm_model = QcuCommModel::kQcuError;
static std::shared_ptr<Communicator> underlying_communicator = nullptr;

QcuCommModel qcu_set_comm_model (QcuCommModel preferred_comm_model) {
    comm_model = preferred_comm_model;
    underlying_communicator = CommunicatorFactory::get_instance(preferred_comm_model);

    // both MPI and NCCL need mpi_rank variable
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

    return preferred_comm_model;
}
std::string get_qcu_comm_status_string(QcuCommStatus status) {
    if (status == QcuCommStatus::kQcuCommSuccess) {
        return "QcuCommSuccess";
    } else if (status == QcuCommStatus::kQcuCommError) {
        return "ERROR: QcuCommError";
    } else if (status == QcuCommStatus::kQcuCommUnimplemented) {
        return "ERROR: QcuCommUnimplemented";
    } else if (status == QcuCommStatus::kQcuCommInvalidDataType) {
        return "ERROR: QcuCommInvalidDataType";
    } else {
        return "ERROR: QcuCommUnknown";
    }
}
}
}