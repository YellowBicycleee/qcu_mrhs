#pragma once

#include "qcd/qcu_dslash.h"

namespace qcu {
class WilsonDslash : public Dslash {

public:
    WilsonDslash(bool if_metric = false) : Dslash() {}

    virtual ~WilsonDslash() noexcept = default;

    virtual void apply(const std::shared_ptr<DslashParam>) override;

    virtual double flops() override;

private:
    void pre_apply(const std::shared_ptr<DslashParam>);

    void post_apply(const std::shared_ptr<DslashParam>);

};

namespace developing {

class WilsonDslash : public Dslash {
public:
    WilsonDslash(bool if_metric = false) : Dslash() {}

    virtual ~WilsonDslash() noexcept = default;

    virtual void apply(const std::shared_ptr<DslashParam>) override;

    virtual double flops() override;

private:
    void pre_apply(const std::shared_ptr<DslashParam>);

    void post_apply(const std::shared_ptr<DslashParam>);
};

}
}