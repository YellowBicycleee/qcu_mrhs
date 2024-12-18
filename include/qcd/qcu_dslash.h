#pragma once

#include "desc/qcu_desc.h"
#include "qcu_public.h"
#include <cuda_runtime.h>
#include <memory>
namespace qcu {

// clang-format off
struct DslashParam {
    bool dagger_flag;
    QcuPrecision dslash_precision;
    int n_color;
    int m_input;
    int parity;
    double kappa;

    void* __restrict__ fermion_in_MRHS;
    void* __restrict__ fermion_out_MRHS;
    void* __restrict__ gauge;
    const QcuLattDesc* __restrict__ latt_desc;
    const QcuProcDesc* __restrict__ proc_desc;
    cudaStream_t stream1;
    cudaStream_t stream2;

    DslashParam(
        bool dagger_flag_,
        QcuPrecision dslash_precision_,
        int n_color_,
        int m_input_,
        int parity_,
        double kappa_,
        void* fermion_in_MRHS_,
        void* fermion_out_MRHS_,
        void* gauge_,
        const QcuLattDesc* latt_desc_,
        const QcuProcDesc* proc_desc_,
        cudaStream_t stream1_ = NULL,
        cudaStream_t stream2_ = NULL)

        : dagger_flag(dagger_flag_),
        dslash_precision(dslash_precision_),
        n_color(n_color_),
        m_input(m_input_),
        parity(parity_),
        kappa(kappa_),
        fermion_in_MRHS(fermion_in_MRHS_),
        fermion_out_MRHS(fermion_out_MRHS_),
        gauge(gauge_),
        proc_desc(proc_desc_),
        latt_desc(latt_desc_),
        stream1(stream1_),
        stream2(stream2_) {}
};

class Dslash {
public:
    Dslash() : operations_cur_(0), time_utilization_cur(0) {}
    virtual ~Dslash() noexcept = default;
    virtual void apply(const std::shared_ptr<DslashParam> dslash_param) = 0;
    virtual double flops() = 0;

protected:
    inline static double operations_total_ = 0.0;
    inline static double time_utilization_total_ = 0.0;

    double operations_cur_;
    double time_utilization_cur;
private:
    cudaEvent_t cuda_event_;
    void pre_apply(const std::shared_ptr<DslashParam>);
    void post_apply(const std::shared_ptr<DslashParam>);

};

}  // namespace qcu