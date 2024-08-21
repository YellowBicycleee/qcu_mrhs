#pragma once
#include "complex/qcu_complex.cuh"
#include <type_traits>

namespace qcu::qcu_blas {

template <typename _Tp, typename = void> // only support for _Tp = Complex<float, double, half>
struct Complex_axpby;                    // result = aX + bY, (X, Y are complex vectors)

template <typename _Float>
struct Complex_axpby<_Float, std::enable_if_t <std::is_same_v<_Float, float>  ||
                                               std::is_same_v<_Float, double> ||
                                               std::is_same_v<_Float, half> > > 
{    
  // Argument type
  struct Complex_axpbyArgument {
    // start_idx 不在外部赋予，而是在内部赋予
    int               single_vec_len;
    int               inc_idx;
    Complex<_Float>*  res;
    Complex<_Float>*  x;
    Complex<_Float>*  y;
    Complex<_Float>*  a;
    Complex<_Float>*  b;
    cudaStream_t      stream;

    Complex_axpbyArgument(
      Complex<_Float>* res,
      Complex<_Float>* a, 
      Complex<_Float>* x,
      Complex<_Float>* b,
      Complex<_Float>* y,
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
