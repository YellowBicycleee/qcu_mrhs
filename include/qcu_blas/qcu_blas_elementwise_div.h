#pragma once
#include <type_traits>
#include <cuda_fp16.h>
namespace qcu::qcu_blas {

template <typename _Tp>
struct ElementwiseDiv {          //  [r1, r2 ... rN] = [x1 / y1, x2 / y2 ... xN / yN]    
  // Argument type
  struct ElementwiseDivArgument {
    const int     vec_len;
    _Tp*          res;
    _Tp*          x;
    _Tp*          y;
    cudaStream_t  stream;

    ElementwiseDivArgument(
      _Tp*          res,
      _Tp*          x,
      _Tp*          y,
      int           vec_len,
      cudaStream_t  stream = nullptr
    ) : res(res),
        x(x),
        y(y),
        vec_len(vec_len),
        stream(stream) {}
  };

  // methods
  void operator () (ElementwiseDivArgument);
};
  
} // namespace qcu::qcu_blas