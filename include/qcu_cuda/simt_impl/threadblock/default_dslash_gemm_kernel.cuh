//
// Created by wjc on 24-10-30.
//

#pragma once

namespace qcu {

namespace qcu_cuda {

namespace simt_impl {

namespace threadblock {

template <typename _FloatType>
class DefaultDslashGemmKernel {
public:
    DefaultDslashGemmKernel(
        const _FloatType* __restrict__ gauge_field, 
        const _FloatType* __restrict__ spinor_field, 
        _FloatType* __restrict__ result);

    void operator()() const;
private:
    const _FloatType* __restrict__ gauge_field_;
    const _FloatType* __restrict__ spinor_field_;
    _FloatType* __restrict__ result_;
};

}

}

}

}
