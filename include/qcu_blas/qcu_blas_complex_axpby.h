#pragma once
#include "complex/qcu_complex.cuh"
#include <type_traits>

namespace qcu::qcu_blas {

template <
    typename ComputeFloat_,
    typename ScaleFloat_ = ComputeFloat_,
    typename = void
> // only support for _Tp = Complex<float, double, half>
struct Complex_axpby;                    // result = aX + bY, (X, Y are complex vectors)

template <
    typename ComputeFloat_,
    typename ScaleFloat_
>
struct Complex_axpby<
    ComputeFloat_,
    ScaleFloat_,
    std::enable_if_t <std::is_same_v<ComputeFloat_, float> || std::is_same_v<ComputeFloat_, double> || std::is_same_v<ComputeFloat_, half> > >
{    
    // Argument type
    struct Complex_axpbyArgument {
    // start_idx 不在外部赋予，而是在内部赋予
    const int         single_vec_len;
    const int         inc_idx;
    Complex<ComputeFloat_>*  res;
    Complex<ComputeFloat_>*  x;
    Complex<ComputeFloat_>*  y;
    Complex<ScaleFloat_>*    a;
    Complex<ScaleFloat_>*    b;
    cudaStream_t      stream;

    Complex_axpbyArgument(
        Complex<ComputeFloat_>* res,
        Complex<ScaleFloat_>* a,
        Complex<ComputeFloat_>* x,
        Complex<ScaleFloat_>* b,
        Complex<ComputeFloat_>* y,
        int              single_vec_len,
        int              inc_idx,
        cudaStream_t     stream = nullptr
    ) : res(res),
        a(a),
        x(x),
        b(b),
        y(y),
        single_vec_len(single_vec_len),
        inc_idx(inc_idx),
        stream(stream) {}
    };

    // methods
    void operator () (Complex_axpbyArgument);
};

}
