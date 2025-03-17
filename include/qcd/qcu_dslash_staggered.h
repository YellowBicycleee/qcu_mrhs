#pragma once

#include "qcd/qcu_dslash.h"
#include "qcu_cuda_config.cuh"
namespace qcu::simt {

// template <typename CudaOp = qcu::config::cudaOp::Simt>
class StaggeredDslash : public Dslash {
public:
    StaggeredDslash(bool if_metric = false) : Dslash() {}

    virtual ~StaggeredDslash() noexcept = default;

    virtual void apply(const std::shared_ptr<DslashParam>) override;

    // virtual double flops() override;

private:
    void pre_apply(const std::shared_ptr<DslashParam>);

    void post_apply(const std::shared_ptr<DslashParam>);

    void apply_ghost_unpack(DslashParam& dslash_param, int ghost_dim);

    void apply_ghost_pack(DslashParam& dslash_param, int ghost_dim);

    static constexpr int Nspin_ = 1;
};

}
