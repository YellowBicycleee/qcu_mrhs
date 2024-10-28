//
// Created by wjc on 24-10-24.
//
#include "nccl.h"
#include "qcu_helper.h"

#include "communication/communicator.h"

namespace qcu {
namespace communication {

QcuCommStatus NcclCommunicator::send(const void* send_buff, size_t count, QcuDataType data_type, int dest/*dest or peer*/,
        void* comm, void* stream, int tag)
{
    return QcuCommStatus::kQcuCommUnimplemented;
}

QcuCommStatus NcclCommunicator::isend(const void* send_buff, size_t count, QcuDataType data_type, int dest, void* comm,
                                     void* stream, int tag, void* request) {
    return QcuCommStatus::kQcuCommUnimplemented;
}

QcuCommStatus NcclCommunicator::recv(void* recv_buf, size_t count, QcuDataType data_type, int source, void* comm,
                                    void* stream, int tag, void* request_or_status) {
    return QcuCommStatus::kQcuCommUnimplemented;
}

QcuCommStatus NcclCommunicator::irecv(void* recv_buf, size_t count, QcuDataType data_type, int source, void* comm,
                                     void* stream, int tag, void* request_or_status) {
    return QcuCommStatus::kQcuCommUnimplemented;
}

QcuCommStatus NcclCommunicator::reduce_sum(const void* send_buf, void* recv_buf, size_t count, QcuDataType data_type,
                                           void* comm, void* stream) {
    return QcuCommStatus::kQcuCommUnimplemented;
}


}
}