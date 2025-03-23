#include <type_traits>

namespace qcu::qcu_blas {

template <
    typename ComputeFloat_,
    typename ScaleFloat_ = ComputeFloat_,
    typename = void
> // only support for _Tp = Complex<float, double, half>
struct Complex_axpbypcz;                    // result = aX + bY + cZ, (X, Y, Z are complex vectors)

template <typename ComputeFloat_, typename ScaleFloat_>
struct Complex_axpbypcz<
    ComputeFloat_,
    ScaleFloat_,
    std::enable_if_t <std::is_same_v<ComputeFloat_, float> || std::is_same_v<ComputeFloat_, double> || std::is_same_v<ComputeFloat_, half> > >
{    
    // Argument type
    struct Complex_axpbypczArgument {
        // start_idx 不在外部赋予，而是在内部赋予
        const int         single_vec_len;
        const int         inc_idx;
        Complex<ComputeFloat_>*  res;
        Complex<ComputeFloat_>*  x;
        Complex<ComputeFloat_>*  y;
        Complex<ComputeFloat_>*  z;
        Complex<ScaleFloat_>*  a;
        Complex<ScaleFloat_>*  b;
        Complex<ScaleFloat_>*  c;
        cudaStream_t      stream;

        Complex_axpbypczArgument(
            Complex<ComputeFloat_>* res,
            Complex<ScaleFloat_>* a,
            Complex<ComputeFloat_>* x,
            Complex<ScaleFloat_>* b,
            Complex<ComputeFloat_>* y,
            Complex<ScaleFloat_>* c,
            Complex<ComputeFloat_>* z,
            int              single_vec_len,
            int              inc_idx,
            cudaStream_t     stream = nullptr)
        : res(res)
        , a(a)
        , x(x)
        , b(b)
        , y(y)
        , c(c)
        , z(z)
        , single_vec_len(single_vec_len)
        , inc_idx(inc_idx)
        ,  stream(stream) {}
    };

    // methods
    void operator () (Complex_axpbypczArgument);
};

} // namespace qcu::qcu_blas