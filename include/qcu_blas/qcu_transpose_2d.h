#pragma once

namespace qcu::qcu_blas {

template <typename _Tp>
struct Transpose2D 
{
  struct Transpose2DParam {
    int m;
    int n;
    void* output;
    void* input;
  };
  Transpose2DParam param;
  Transpose2D(Transpose2DParam in_param) noexcept : param(in_param) {}
  Transpose2D(int m, int n, void* out, void* in) noexcept
    : param({m, n, out, in}) {}

  void operator() ();
  ~Transpose2D() noexcept = default;
};

}