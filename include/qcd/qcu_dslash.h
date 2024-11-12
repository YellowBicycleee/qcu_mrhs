#pragma once

#include "desc/qcu_desc.h"
#include "qcu_public.h"
#include <cuda_runtime.h>

namespace qcu {

// clang-format off
struct DslashParam {
    bool daggerFlag;
    QcuPrecision precision;
    int nColor;
    int mInput;
    int parity;
    double kappa;

    void* fermionIn_MRHS;
    void* fermionOut_MRHS;
    void* gauge;
    const QcuLattDesc* lattDesc;
    const QcuProcDesc* procDesc;
    cudaStream_t stream1;
    cudaStream_t stream2;

    DslashParam(bool p_daggerFlag,
                QcuPrecision p_precision,
                int p_nColor, 
                int p_mInput, 
                int p_parity,
                double p_kappa, 
                void* p_fermionIn_MRHS,
                void* p_fermionOut_MRHS,
                void* p_gauge, 
                const QcuLattDesc* p_lattDesc,
                const QcuProcDesc* p_procDesc,
                cudaStream_t p_stream1 = NULL,
                cudaStream_t p_stream2 = NULL)
        : daggerFlag(p_daggerFlag),
          precision(p_precision),
          nColor(p_nColor),
          mInput(p_mInput),
          parity(p_parity),
          kappa(p_kappa),
          fermionIn_MRHS(p_fermionIn_MRHS),
          fermionOut_MRHS(p_fermionOut_MRHS),
          gauge(p_gauge),
          procDesc(p_procDesc),
          lattDesc(p_lattDesc) , 
          stream1(p_stream1),
          stream2(p_stream2) {}
};

class Dslash {
protected:
    bool if_metric_ = false;
    double operations_ = 0.0;
    double time_ = 0.0;

    virtual void pre_apply(const DslashParam&) = 0;
    virtual void post_apply(const DslashParam&) = 0;
public:
    Dslash(bool if_matric = false) : if_metric_(if_matric) {}
    virtual ~Dslash() noexcept = default;
    virtual void apply(DslashParam param) = 0;
    virtual double flops() = 0;
};

}  // namespace qcu