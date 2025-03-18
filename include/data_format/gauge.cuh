#pragma once

#include "qcu_public.h"

namespace qcu::base {




class Field {
public:
    Field(
        bool use_external,
        bool use_even_odd_precondition,
        const QcuPrecision precision,
        void* const data
    )
    : use_external_(use_external)
    , use_even_odd_precondition_(use_even_odd_precondition)
    , precision_(precision)
    , data_(data)
    {}

    ~Field() = default;

    void* data() const {
        return data_;
    }
    bool assert_precision(QcuPrecision precision) const {
        return precision_ == precision;
    }

private:
    const bool use_external_; // it just points to the external memory, it does not own the memory
    const bool use_even_odd_precondition_;
    const QcuPrecision precision_;
    void* const data_;
};

}