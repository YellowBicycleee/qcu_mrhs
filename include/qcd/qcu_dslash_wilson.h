#pragma once

#include "qcd/qcu_dslash.h"

namespace qcu {
class WilsonDslash : public Dslash {
    virtual void pre_apply(const DslashParam&);
    virtual void post_apply(const DslashParam&);
public:
    WilsonDslash(bool if_metric = false) : Dslash(if_metric) {}
    virtual ~WilsonDslash() noexcept = default;
    // virtual void async_work_flow();

    virtual void apply(DslashParam dslashParam);
    virtual double flops();
};

}