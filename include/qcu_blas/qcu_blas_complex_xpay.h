#include <type_traits>
#include <cuda_fp16.h>
namespace qcu::qcu_blas {

template <typename ComputeFloat_, typename ScaleFloat_ = ComputeFloat_, typename = void>
struct Complex_xpay;

template <typename ComputeFloat_, typename ScaleFloat_>
struct Complex_xpay<
    ComputeFloat_,
    ScaleFloat_,
    std::enable_if_t <std::is_same_v<ComputeFloat_, float> || std::is_same_v<ComputeFloat_, double> || std::is_same_v<ComputeFloat_, half> > >
{
  // Argument type
    struct Complex_xpayArgument {
        // start_idx 不在外部赋予，而是在内部赋予
        const int         single_vec_len;
        const int         inc_idx;
        Complex<ComputeFloat_>*  res;
        Complex<ComputeFloat_>*  x;
        Complex<ComputeFloat_>*  y;
        Complex<ScaleFloat_>*    a;
        cudaStream_t      stream;

        Complex_xpayArgument(
            Complex<ComputeFloat_>* res,
            Complex<ComputeFloat_>* x,
            Complex<ScaleFloat_>*   a,
            Complex<ComputeFloat_>* y,
            int              single_vec_len,
            int              inc_idx,
            cudaStream_t     stream = nullptr
        ) : res(res),
            x(x),
            a(a),
            y(y),
            single_vec_len(single_vec_len),
            inc_idx(inc_idx),
            stream(stream) {}
        };

        // methods
        void operator () (Complex_xpayArgument);
    };

} // namespace qcu::qcu_blas