#pragma once
#include <type_traits>

namespace qcu::qcu_blas {

template <typename _Tp>
struct ElementwiseInit {          //  [r1, r2 ... rN] = [x1 / y1, x2 / y2 ... xN / yN]    
  // Argument type
  struct ElementwiseInitArgument {
    int           vec_len;
    _Tp*          res;
    _Tp           val;
    cudaStream_t  stream;

    ElementwiseInitArgument(
      _Tp*          res,
      _Tp           val,
      int           vec_len,
      cudaStream_t  stream = nullptr
    ) : res(res),
        val(val),
        vec_len(vec_len),
        stream(stream) {}
  };

  // methods
  void operator () (ElementwiseInitArgument);
};
  
} // namespace qcu::qcu_blas