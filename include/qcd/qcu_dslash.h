#pragma once

#include "desc/qcu_desc.h"
#include "qcu.h"
#include "qcu_public.h"
#include <cuda.h>
#include <cuda_runtime.h>
namespace qcu {

// clang-format off
struct DslashParam {
    bool daggerFlag;
    QCU_PRECISION precision;
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
                QCU_PRECISION p_precision,
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

// clang-format on
class Dslash {
   protected:
    // DslashParam& dslashParam_;
    DslashParam* dslashParam_;
    float dslashFlops_;

   public:
    // Dslash(DslashParam& dslashParam) : dslashParam_(dslashParam) {}
    Dslash() = delete;
    Dslash(DslashParam* dslashParam) : dslashParam_(dslashParam) {}
    void setParam(DslashParam* dslashParam) { dslashParam_ = dslashParam; }
    virtual ~Dslash() = default;
    virtual void apply() = 0;
    virtual void preApply() = 0;
    virtual void postApply() = 0;
    virtual void flops() = 0;
};

class WilsonDslash : public Dslash {
   public:
    WilsonDslash(DslashParam* dslashParam) : Dslash(dslashParam) {}
    virtual ~WilsonDslash() = default;
    virtual void apply();
    virtual void preApply();
    virtual void postApply();
    virtual void flops();
};

}  // namespace qcu