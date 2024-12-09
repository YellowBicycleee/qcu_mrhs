#pragma once

#include "qcu_helper.h"

namespace qcu::gemm {

/// Shape of a matrix multiply-add operation
template <
  /// Rows of matrix product
  int M = 1,
  /// Columns of matrix product
  int N = 1
>
struct MatShape {
  static int constexpr kM = M;
  static int constexpr kN = N;

  static int constexpr kMN = M * N;

  QCU_DEVICE static int atM () { return kM; }
  QCU_DEVICE static int atN () { return kN; }
};

/// Type alias of the transpose of a GemmShape
template <
  /// concept: GemmShape
  typename Shape
>
using MatShapeTranspose = MatShape<Shape::kN, Shape::kM>;


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

/// @brief namespace for Matrix Major
namespace matrix_major {

/**
 * @brief Matrix Major
 * @tparam transpose_flag_ true for row major, false for col major
 */
template <bool transpose_flag_> // transpose for row major, non-transpose for col major
struct MatrixMajor {
    static constexpr bool if_transpose = transpose_flag_; 
};

/// @brief Row Major
using MatrixRowMajor = MatrixMajor<true>;
/// @brief Col Major
using MatrixColMajor = MatrixMajor<false>;

/// @brief Matrix Major transpose
template <typename MatrixMajor_>
using MatrixMajorTranspose = MatrixMajor<!MatrixMajor_::if_transpose>;
}

/**
 * @brief namespace for Mma Type
 *  SimtOp version donnot use tensor core
 *  TensorOp version use tensor core
 */
namespace mma_type {

template <bool use_tensor_core_flag>
struct MmaType {
    static constexpr bool use_tensor_core = use_tensor_core_flag;
};
/// @brief SimtOp version donnot use tensor core
using GemmSimtOp = MmaType<false>;
/// @brief TensorOp version use tensor core
using GemmTensorOp = MmaType<true>;
}

}