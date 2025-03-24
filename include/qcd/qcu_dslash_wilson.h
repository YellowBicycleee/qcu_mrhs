#pragma once

#include "qcd/qcu_dslash.h"

namespace qcu {
namespace tensorop {
class WilsonDslash : public Dslash {

public:
    WilsonDslash(bool if_metric = false) : Dslash() {}

    virtual ~WilsonDslash() noexcept = default;

    virtual void apply(const std::shared_ptr<DslashParam>) override;


protected:
    static constexpr int Nspin_ = 4;

    inline static double flop_ = 0.0;

    inline static double time_ = 0.0;

private:
    void pre_apply(const std::shared_ptr<DslashParam>);

    void post_apply(const std::shared_ptr<DslashParam>);

    void apply_ghost_unpack(DslashParam& dslash_param, int ghost_dim);

    // void apply_ghost_pack(DslashParam& dslash_param, int ghost_dim);


};
}

namespace simt {

class WilsonDslash : public Dslash {
public:
    WilsonDslash(bool if_metric = false) : Dslash() {}

    virtual ~WilsonDslash() noexcept = default;

    virtual void apply(const std::shared_ptr<DslashParam>) override;

protected:
    static constexpr int Nspin_ = 4;

    inline static double flop_ = 0.0;

    inline static double time_ = 0.0;

private:
    void pre_apply(const std::shared_ptr<DslashParam>);

    void post_apply(const std::shared_ptr<DslashParam>);

    // void apply_ghost_unpack(DslashParam& dslash_param, int ghost_dim);

    // void apply_ghost_pack(DslashParam& dslash_param, int ghost_dim);
};

}
}