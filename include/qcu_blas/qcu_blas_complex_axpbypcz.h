#include <type_traits>

namespace qcu::qcu_blas {

template <typename _Tp, typename = void> // only support for _Tp = Complex<float, double, half>
struct Complex_axpbypcz;                    // result = aX + bY + cZ, (X, Y, Z are complex vectors)

template <typename _Float>
struct Complex_axpbypcz<_Float, std::enable_if_t <std::is_same_v<_Float, float>  ||
                                                  std::is_same_v<_Float, double> ||
                                                  std::is_same_v<_Float, half> > > 
{    
  // Argument type
  struct Complex_axpbypczArgument {
    // start_idx 不在外部赋予，而是在内部赋予
    int               single_vec_len;
    int               inc_idx;
    Complex<_Float>*  res;
    Complex<_Float>*  x;
    Complex<_Float>*  y;
    Complex<_Float>*  z;
    Complex<_Float>*  a;
    Complex<_Float>*  b;
    Complex<_Float>*  c;
    cudaStream_t      stream;

    Complex_axpbypczArgument(
      Complex<_Float>* res,
      Complex<_Float>* a, 
      Complex<_Float>* x,
      Complex<_Float>* b,
      Complex<_Float>* y,
      Complex<_Float>* c,
      Complex<_Float>* z,
      int              single_vec_len,
      int              inc_idx,
      cudaStream_t     stream
    ) : res(res),
        a(a),
        x(x),
        b(b),
        y(y),
        c(c),
        z(z),
        single_vec_len(single_vec_len),
        inc_idx(inc_idx),
        stream(stream) {}
  };

  // methods
  void operator () (Complex_axpbypczArgument);
};

} // namespace qcu::qcu_blas