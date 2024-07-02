#pragma once

#include <assert.h>

// #include "qcu_macro.cuh"
#include "qcu_enum.h"
#include "qcu_macro.h"
// clang-format off
class Point {
   private:
    int x_;
    int y_;
    int z_;
    int t_;
    int parity_;

   public:
    Point() = default;
    __device__ __forceinline__ Point(const Point &rhs) : x_(rhs.x_), y_(rhs.y_), z_(rhs.z_), t_(rhs.t_), parity_(rhs.parity_) {}
    __device__ __forceinline__ Point(int x, int y, int z, int t, int parity) : x_(x), y_(y), z_(z), t_(t), parity_(parity) {}

    __device__ __forceinline__ int getParity() const { return parity_; }
    __device__ __forceinline__ Point move(int front_back, int dim, int Lx, int Ly, int Lz, int Lt) const {
        // 1-front 0-back
        // assert(abs(dim) >= X_DIM && abs(dim) < Nd);
        // assert(front_back == BWD || front_back == FWD);

        int new_pos;
        int eo = (y_ + z_ + t_) & 0x01;  // (y+z+t)%2

        if (dim == X_DIM) {
            if (!front_back)  // front_back == BWD
            {
                new_pos = x_ + (eo == parity_) * (-1 + (x_ == 0) * Lx);
                return Point(new_pos, y_, z_, t_, 1 - parity_);
            } else  // front_back == FWD
            {
                new_pos = x_ + (eo != parity_) * (1 + (x_ == Lx - 1) * (-Lx));
                return Point(new_pos, y_, z_, t_, 1 - parity_);
            }
        } else if (dim == Y_DIM) {
            if (!front_back)  // front_back == BWD
            {
                new_pos = y_ - 1 + (y_ == 0) * Ly;
                return Point(x_, new_pos, z_, t_, 1 - parity_);
            } else  // front_back == FWD
            {
                new_pos = y_ + 1 + (y_ == Ly - 1) * (-Ly);
                return Point(x_, new_pos, z_, t_, 1 - parity_);
            }
        } else if (dim == Z_DIM) {
            if (!front_back) {
                new_pos = z_ - 1 + (z_ == 0) * Lz;
                return Point(x_, y_, new_pos, t_, 1 - parity_);
            } else {
                new_pos = z_ + 1 + (z_ == Lz - 1) * (-Lz);
                return Point(x_, y_, new_pos, t_, 1 - parity_);
            }
        } else if (dim == T_DIM) {
            if (!front_back) {
                new_pos = t_ - 1 + (t_ == 0) * Lt;
                return Point(x_, y_, z_, new_pos, 1 - parity_);
            } else {
                new_pos = t_ + 1 + (t_ == Lt - 1) * (-Lt);
                return Point(x_, y_, z_, new_pos, 1 - parity_);
            }
        } else {
            return *this;
        }
    }

    template <typename Float>
    __device__ __forceinline__ Float* getGaugeAddr (Float* base, int dim, int half_Lx, int Ly, int Lz, int Lt, int n_color) const {
        return base + 2 * (((dim * 2 + parity_) * half_Lx * Ly * Lz * Lt + IDX4D(t_, z_, y_, x_, Lz, Ly, half_Lx))) * n_color * n_color;
    }
    template <typename Float>
    __device__ __forceinline__ Float* getGatheredColorSpinorAddr (Float* base_pc, int half_Lx, int Ly, int Lz, int Lt, int n_color, int m_input) const {
        return base_pc + 2 * (IDX4D(t_, z_, y_, x_, Lz, Ly, half_Lx) * m_input * Ns * n_color);
    }
    

    __device__ __forceinline__ Point &operator=(const Point &rhs) {
        x_ = rhs.x_;
        y_ = rhs.y_;
        z_ = rhs.z_;
        t_ = rhs.t_;
        parity_ = rhs.parity_;
        return *this;
    }
};