#pragma once

#include <cuda_runtime.h>
#include <qcu_helper_macro.h>

#include <memory>
#include "timer/timer.h"
#include "data_format/fermion.cuh"
#include "desc/qcu_desc.h"
#include "qcu_public.h"
#include "qcu_extern_macro.h"

namespace qcu {

// clang-format off
struct DslashParam {
    bool dagger_flag;
    QcuPrecision dslash_precision;
    QcuStaggeredPhase staggered_phase; // = QcuStaggeredPhase::kQcuStaggeredPhaseNo;  // add attribute
    int t_boudary = 1;                                                                  // add attribute
    int n_color;
    int m_input;
    int parity;
    double kappa;
    void* __restrict__ fermion_in_MRHS;
    void* __restrict__ fermion_out_MRHS;
    void* __restrict__ gauge;
    const QcuLattDesc* __restrict__ latt_desc;
    const QcuProcDesc* __restrict__ proc_desc;
    std::vector<cudaStream_t>& streams;
    std::shared_ptr<qcu::FermionGhost<Nd>> fermion_ghost;
    DslashParam(
        bool dagger_flag_,
        QcuPrecision dslash_precision_,
        QcuStaggeredPhase staggered_phase_,
        int t_boudary_,
        int n_color_,
        int m_input_,
        int parity_,
        double kappa_,
        void* fermion_in_MRHS_,
        void* fermion_out_MRHS_,
        void* gauge_,
        const QcuLattDesc* latt_desc_,
        const QcuProcDesc* proc_desc_,
        std::vector<cudaStream_t>& streams_,
        std::shared_ptr<qcu::FermionGhost<Nd>> fermion_ghost_ = nullptr
        )

    : dagger_flag(dagger_flag_)
    , dslash_precision(dslash_precision_)
    , staggered_phase(staggered_phase_)
    , t_boudary(t_boudary_)
    , n_color(n_color_)
    , m_input(m_input_)
    , parity(parity_)
    , kappa(kappa_)
    , fermion_in_MRHS(fermion_in_MRHS_)
    , fermion_out_MRHS(fermion_out_MRHS_)
    , gauge(gauge_)
    , proc_desc(proc_desc_)
    , latt_desc(latt_desc_)
    , streams(streams_)
    , fermion_ghost(fermion_ghost_)
    {}
};

class Dslash {
public:
    Dslash() : timer_() {}

    virtual ~Dslash() noexcept = default;

    virtual void apply(const std::shared_ptr<DslashParam> dslash_param) = 0;

protected:

    inline static double flop_ = 0.0;

    inline static double time_ = 0.0;

    double flop_per_rhs_ = 0.0;

    qcu::perf::Timer timer_;

private:
    void pre_apply(const std::shared_ptr<DslashParam>);

    void post_apply(const std::shared_ptr<DslashParam>);
};

}  // namespace qcu