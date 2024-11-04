#pragma once

#include <desc/qcu_desc.h>

#include <cstdint>

#include "lattice_desc.h"
#include "qcu_helper.h"
namespace qcu {

namespace device {

template <
    typename _Tp // element type
>
class DslashWilsonDevice {
public:
    struct Argument {
        bool dagger_flag;
        bool mat_flag;      // with or without mat

        int32_t parity;     // parity of fermion out, contrast with fermion in
        int32_t n_color;
        int32_t m_rhs;
        // dim length
        qcu::QcuLattDesc* latt_desc;
        qcu::QcuProcDesc* proc_desc;

        _Tp* output; // fermion out
        _Tp* input;  // fermion in   
        _Tp* gauge;  // gauge
        _Tp kappa;
    };
    
private:
    

public:
    DslashWilsonDevice () = default;

    qcu::QcuStatus run (cudaStream_t stream = nullptr) {
        return qcu::QcuStatus::kErrorInternal;
    }

    qcu::QcuStatus operator () (cudaStream_t stream = nullptr) {
        return run (stream);
    }
};

}

}