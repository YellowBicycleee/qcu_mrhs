//
// Created by wjc on 24-10-24.
//

#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
namespace qcu {
namespace communication {
/* Communication model */
enum class QcuCommModel {
    kQcuCommMpi,
    kQcuCommNccl,
    kQcuError
  };
/* Data types */
enum QcuDataType{
    kQcuInt8    = 0, kQcuChar       = 0,
    kQcuUint8   = 1, kQcuInt32      = 2,
    kQcuInt     = 2, kQcuUint32     = 3,
    kQcuInt64   = 4, kQcuUint64     = 5,
    kQcuFloat16 = 6, kQcuHalf       = 6,
    kQcuFloat32 = 7, kQcuFloat      = 7,
    kQcuFloat64 = 8, kQcuDouble     = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__)
    kQcuBfloat16 = 9, kQcuNumTypes   = 10
#else
    kQcuNumTypes   = 9
#endif
} ;

enum class QcuCommStatus {
    kQcuCommSuccess = 0,
    kQcuCommError = 1,
    kQcuCommUnimplemented = 2,
    kQcuCommInvalid = 3,
};

class Communicator {
public:
    //               MPI_Send (const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
    //               MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
    // ncclResult_t  ncclSend (const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
    virtual QcuCommStatus send(const void* send_buff, size_t count, QcuDataType data_type, int dest/*dest or peer*/,
        void* comm, void* stream, int tag) = 0;
    virtual QcuCommStatus isend(const void* send_buff, size_t count, QcuDataType data_type, int dest,
        void* comm, void* stream, int tag, void * request) = 0;

    // int           MPI_Recv (void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
    // int           MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
    // ncclResult_t  ncclRecv (void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
    virtual QcuCommStatus recv(void* recv_buf, size_t count, QcuDataType data_type, int source, void* comm,
        void* stream, int tag, void * request_or_status) = 0;
    virtual QcuCommStatus irecv(void* recv_buf, size_t count, QcuDataType data_type, int source, void* comm,
        void* stream, int tag, void * request_or_status) = 0;

    // reduce_sum
    // int
    //     MPI_Allreduce(const void *sendbuf, void *recvbuf,   int count,    MPI_Datatype datatype,   MPI_Op op, MPI_Comm comm);
    // ncclResult_t
    //     ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
    virtual QcuCommStatus reduce_sum (const void* send_buf, void* recv_buf, size_t count, QcuDataType data_type,
        void* comm, void* stream) = 0;

    // barrier
    virtual QcuCommStatus barrier() = 0;
};

class MpiCommunicator : public Communicator {
public:
    QcuCommStatus send(const void* send_buff, size_t count, QcuDataType data_type, int dest/*dest or peer*/,
        void* comm, void* stream, int tag) override;
    QcuCommStatus isend(const void* send_buff, size_t count, QcuDataType data_type, int dest,
        void* comm, void* stream, int tag, void * request) override;

    QcuCommStatus recv(void* recv_buf, size_t count, QcuDataType data_type, int source, void* comm,
        void* stream, int tag, void * request_or_status) override;
    QcuCommStatus irecv(void* recv_buf, size_t count, QcuDataType data_type, int source, void* comm,
        void* stream, int tag, void * request_or_status) override;

    QcuCommStatus reduce_sum (const void* send_buf, void* recv_buf, size_t count, QcuDataType data_type,
        void* comm, void* stream) override;

    // barrier
    QcuCommStatus barrier() override;
};

class NcclCommunicator : public Communicator {
public:
    QcuCommStatus send(const void* send_buff, size_t count, QcuDataType data_type, int dest/*dest or peer*/,
    void* comm, void* stream, int tag) override;
    QcuCommStatus isend(const void* send_buff, size_t count, QcuDataType data_type, int dest,
        void* comm, void* stream, int tag, void * request) override;

    QcuCommStatus recv(void* recv_buf, size_t count, QcuDataType data_type, int source, void* comm,
        void* stream, int tag, void * request_or_status) override;
    QcuCommStatus irecv(void* recv_buf, size_t count, QcuDataType data_type, int source, void* comm,
        void* stream, int tag, void * request_or_status) override;

    QcuCommStatus reduce_sum (const void* send_buf, void* recv_buf, size_t count, QcuDataType data_type,
        void* comm, void* stream) override;

    // nccl does not support barrier
    QcuCommStatus barrier() override {
        return QcuCommStatus::kQcuCommSuccess;
    }
};

// Factory
class MpiCommunicatorFactory {
public:
    static std::shared_ptr<MpiCommunicator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<MpiCommunicator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<MpiCommunicator> instance;
};
class NcclCommunicatorFactory {
public:
    static std::shared_ptr<NcclCommunicator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<NcclCommunicator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<NcclCommunicator> instance;
};

class CommunicatorFactory {
public:
    static std::shared_ptr<Communicator> get_instance (QcuCommModel comm_model) {
        switch (comm_model) {
            case QcuCommModel::kQcuCommMpi: {
                return MpiCommunicatorFactory::get_instance();
            }
            case QcuCommModel::kQcuCommNccl: {
                return NcclCommunicatorFactory::get_instance();
            }
            default: {
                assert(0);
            }
            return nullptr;
        }
    }
};

}
}