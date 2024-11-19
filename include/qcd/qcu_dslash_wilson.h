#pragma once

#include "qcd/qcu_dslash.h"

namespace qcu {
class WilsonDslash : public Dslash {
    virtual void pre_apply(const std::shared_ptr<DslashParam>) override;
    virtual void post_apply(const std::shared_ptr<DslashParam>) override;
public:
    WilsonDslash(bool if_metric = false) : Dslash(if_metric) {}
    virtual ~WilsonDslash() noexcept = default;
    // virtual void async_work_flow();

    virtual void apply(const std::shared_ptr<DslashParam>) override;
    virtual double flops() override;
};

namespace developing {

class WilsonDslash : public Dslash {
    virtual void pre_apply(const std::shared_ptr<DslashParam>) override;
    virtual void post_apply(const std::shared_ptr<DslashParam>) override;
public:
    WilsonDslash(bool if_metric = false) : Dslash(if_metric) {}
    virtual ~WilsonDslash() noexcept = default;
    // virtual void async_work_flow();

    virtual void apply(const std::shared_ptr<DslashParam>);
    virtual double flops();
};

}
}