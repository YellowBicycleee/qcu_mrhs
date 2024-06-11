#pragma once

#include "desc/qcu_desc.h"
#include "qcu.h"
#include "qcu_enum.h"

namespace qcu {

// clang-format off
struct DslashParam {
    int nColor;
    int nInput;
    double kappa;
    int parity;
    int daggerFlag;

    void* fermionIn_MRHS;
    void* fermionOut_MRHS;
    void* gauge;
    const QcuLattDesc& lattDesc;
    const QcuProcDesc& procDesc;

    DslashParam(int p_nColor, 
                int p_nInput, 
                double p_kappa, 
                int p_parity,
                int p_daggerFlag, 
                void* p_fermionIn_MRHS,
                void* p_fermionOut_MRHS,
                void* p_gauge, 
                const QcuLattDesc& p_lattDesc,
                const QcuProcDesc& p_procDesc
            )
        : nColor(p_nColor),
          nInput(p_nInput),
          kappa(p_kappa),
          parity(p_parity),
          daggerFlag(p_daggerFlag),
          fermionIn_MRHS(p_fermionIn_MRHS),
          fermionOut_MRHS(p_fermionOut_MRHS),
          gauge(p_gauge),
          procDesc(p_procDesc),
          lattDesc(p_lattDesc) {}
};

// clang-format on
class Dslash {
   protected:
    DslashParam& dslashParam_;
    float dslashFlops_;

   public:
    Dslash(const DslashParam& dslashParam) : dslashParam_(dslashParam) {}
    virtual ~Dslash() {}
    virtual void apply() = 0;
    virtual void preApply() = 0;
    virtual void postApply() = 0;
    virtual void flops() = 0;
};

class WilsonDslash : public Dslash {
   public:
    WilsonDslash(const DslashParam& dslashParam) : Dslash(dslashParam) {}
    virtual ~WilsonDslash() {}
    virtual void apply();
    virtual void preApply();
    virtual void postApply();
    // TODO : calc flops
    virtual void flops() {}
};

}  // namespace qcu