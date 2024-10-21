#pragma once

#include <cstdint>

#include "lattice_desc.h"

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
        Latt_Desc latt_desc;

        _Tp* output; // fermion out
        _Tp* input;  // fermion in   
        _Tp* gauge;  // gauge
        _Tp kappa;
    };
    
private:
    

public:
    DslashWilsonDevice () = default;

    Status run (cudaStream_t stream = nullptr) {
        throw int;
    }

    Status operator () (cudaStream_t stream = nullptr) {
        return run (stream);
    }
};

}

}