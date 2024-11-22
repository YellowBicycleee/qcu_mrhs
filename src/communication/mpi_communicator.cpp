//
// Created by wjc on 24-10-24.
//
#include <mpi.h>
#include "qcu_helper.h"
#include "check_error/check_mpi.h"
#include "communication/communicator.h"

namespace qcu {
namespace communication {

static MPI_Datatype get_mpi_data_type(QcuDataType data_type) {
    MPI_Datatype mpi_data_type;
    if (data_type == QcuDataType::kQcuDouble) {
        mpi_data_type = MPI_DOUBLE;
    } else if (data_type == QcuDataType::kQcuFloat) {
        mpi_data_type = MPI_FLOAT;
    } else if (data_type == QcuDataType::kQcuInt) {
        mpi_data_type = MPI_INT;
    } else {
        mpi_data_type = MPI_DATATYPE_NULL;
    }
    return mpi_data_type;
}

QcuCommStatus MpiCommunicator::send (const void* send_buff, size_t count, QcuDataType data_type, int dest,
    int tag, void* comm, [[maybe_unused]] void* stream)
{
    MPI_Datatype mpi_data_type = get_mpi_data_type(data_type);
    if (mpi_data_type == MPI_DATATYPE_NULL) {
        return QcuCommStatus::kQcuCommInvalidDataType;
    }
    CHECK_MPI(MPI_Send(send_buff, (int)count, mpi_data_type, dest, tag, static_cast<MPI_Comm>(comm)));
    return QcuCommStatus::kQcuCommSuccess;
}

QcuCommStatus MpiCommunicator::isend(const void* send_buff, size_t count, QcuDataType data_type, int dest,
    int tag, void* comm, void * request, [[maybe_unused]] void* stream)
{
    MPI_Datatype mpi_data_type = get_mpi_data_type(data_type);
    if (mpi_data_type == MPI_DATATYPE_NULL) {
            return QcuCommStatus::kQcuCommInvalidDataType;
    }
    MPI_Isend(send_buff, count, mpi_data_type, dest, tag, static_cast<MPI_Comm>(comm),
        static_cast<MPI_Request*>(request));
    return QcuCommStatus::kQcuCommSuccess;
}

QcuCommStatus MpiCommunicator::recv(void* recv_buf, QcuDataType data_type, size_t count, int source, int tag, void* comm,
    void * request_or_status, [[maybe_unused]] void* stream)
{
    MPI_Datatype mpi_data_type = get_mpi_data_type(data_type);
    if (mpi_data_type == MPI_DATATYPE_NULL) {
        return QcuCommStatus::kQcuCommInvalidDataType;
    }
    CHECK_MPI(MPI_Recv(recv_buf, (int)count, mpi_data_type, source, tag, static_cast<MPI_Comm>(comm),
        static_cast<MPI_Status*>(request_or_status)));
    return QcuCommStatus::kQcuCommSuccess;
}

QcuCommStatus MpiCommunicator::irecv(void* recv_buf, QcuDataType data_type, size_t count, int source, int tag, void* comm,
    void * request_or_status, [[maybe_unused]] void* stream)
{
    MPI_Datatype mpi_data_type = get_mpi_data_type(data_type);
    if (mpi_data_type == MPI_DATATYPE_NULL) {
        return QcuCommStatus::kQcuCommInvalidDataType;
    }
    CHECK_MPI(MPI_Irecv(recv_buf, count, mpi_data_type, source, tag, static_cast<MPI_Comm>(comm),
        static_cast<MPI_Request*>(request_or_status)));
    return  QcuCommStatus::kQcuCommSuccess;
}

QcuCommStatus MpiCommunicator::reduce_sum (const void* send_buf, void* recv_buf, size_t count, QcuDataType data_type,
    void* comm, [[maybe_unused]] void* stream)
{
    MPI_Datatype mpi_data_type = get_mpi_data_type(data_type);
    if (mpi_data_type == MPI_DATATYPE_NULL) {
            return QcuCommStatus::kQcuCommInvalidDataType;
    }
    CHECK_MPI(MPI_Allreduce(send_buf, recv_buf, count, mpi_data_type, MPI_SUM, static_cast<MPI_Comm>(comm)));
    return QcuCommStatus::kQcuCommSuccess;
}

// barrier
QcuCommStatus MpiCommunicator::barrier(void* comm) {
    CHECK_MPI(MPI_Barrier(static_cast<MPI_Comm>(comm)));
    return QcuCommStatus::kQcuCommSuccess;
}

}
}