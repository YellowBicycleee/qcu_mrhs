#pragma once

#include "qcu_helper.h"

namespace qcu::gemm {

/// Shape of a matrix multiply-add operation
template <
  /// Rows of matrix product
  int M = 1,
  /// Columns of matrix product
  int N = 1,
  /// Inner dimension of matrix product
  int K = 1
>
struct GemmShape {
  static int constexpr kM = M;
  static int constexpr kN = N;
  static int constexpr kK = K;

  static int constexpr kMN = M * N;
  static int constexpr kMK = M * K;
  static int constexpr kKN = N * K;
  static int constexpr kMNK = M * N * K;

  static int constexpr kCount = kMNK;

  QCU_DEVICE static int atM () { return kM; }
  QCU_DEVICE static int atN () { return kN; }
  QCU_DEVICE static int atK () { return kK; }
};

/// Type alias of the transpose of a GemmShape
template <
  /// concept: GemmShape
  typename Shape
>
using GemmShapeTranspose = GemmShape<Shape::kN, Shape::kM, Shape::kK>;

}