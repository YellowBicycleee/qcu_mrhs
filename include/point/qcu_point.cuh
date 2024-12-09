#pragma once

#include "qcu_helper.h"
#include "qcu_public.h"
#include "desc/qcu_desc.h"
// even odd preconditioned Coord
class Point {
public:
    Point () = default;
    QCU_DEVICE Point (int x, int y, int z, int t, int parity) : dims{x, y, z, t}, parity_(parity) {}
    Point (const Point&) = default;
    Point &operator=(const Point &rhs) = default;

    QCU_DEVICE Point move(int dir, int dim, int half_Lx, int Ly, int Lz, int Lt) const {
        // 1-front 0-back

        int new_pos;
        int eo = (Y() + Z() + T()) & 0x01;  // (y+z+t)%2

        if (dim == X_DIM) {
            if (BWD == dir)  // front_back == BWD
            {
                new_pos = X() + (eo == parity_) * (-1 + (X() == 0) * half_Lx);
                return Point(new_pos, Y(), Z(), T(), 1 - parity_);
            } else  // front_back == FWD
            {
                new_pos = X() + (eo != parity_) * (1 + (X() == half_Lx - 1) * (-half_Lx));
                return Point(new_pos, Y(), Z(), T(), 1 - parity_);
            }
        } else if (dim == Y_DIM) {
            if (BWD == dir)  // front_back == BWD
            {
                new_pos = Y() - 1 +  (Y() == 0) * Ly;
                return Point(X(), new_pos, Z(), T(), 1 - parity_);
            } else  // front_back == FWD
            {
                new_pos = Y() + 1 +  (Y() == Ly - 1) * (-Ly);
                return Point(X(), new_pos, Z(), T(), 1 - parity_);
            }
        } else if (dim == Z_DIM) {
            if (BWD == dir) {
                new_pos = Z() - 1 +  (Z() == 0) * Lz;
                return Point(X(), Y(), new_pos, T(), 1 - parity_);
            } else {
                new_pos = Z() + 1 +  (Z() == Lz - 1) * (-Lz);
                return Point(X(), Y(), new_pos, T(), 1 - parity_);
            }
        } else if (dim == T_DIM) {
            if (BWD == dir) {
                new_pos = T() - 1 + (T() == 0) * Lt;
                return Point(X(), Y(), Z(), new_pos, 1 - parity_);
            } else {
                new_pos = T() + 1 + (T() == Lt - 1) * (-Lt);
                return Point(X(), Y(), Z(), new_pos, 1 - parity_);
            }
        } else {
            return *this;
        }
    }

    QCU_DEVICE Point move(int dir, int dim, qcu::QcuLattDesc &half_desc) const {
        return move(dir, dim, half_desc.X(), half_desc.Y(), half_desc.Z(), half_desc.T());
    }
    // be careful : there is Float instead of Float2
    template <typename Float>
    QCU_DEVICE Float* getGaugeAddr (Float* base, int dim, int half_Lx, int Ly, int Lz, int Lt, int n_color) const {
        return base + 2 * ((dim * 2 + parity_) * half_Lx * Ly * Lz * Lt + get_1d_idx(half_Lx, Ly, Lz)) * n_color * n_color;
    }
    template <typename Float>
    QCU_DEVICE Float* getGaugeAddr (Float* base, int dim, qcu::QcuLattDesc &half_desc, int n_color) const {
        return getGaugeAddr(base, dim, half_desc.X(), half_desc.Y(), half_desc.Z(), half_desc.T(), n_color);
    }

    template <typename Float>
    QCU_DEVICE Float* getGatheredColorSpinorAddr (  Float* base_pc, int half_Lx, int Ly, int Lz, int Lt,
                                                    int n_color, int m_input) const
    {
        return base_pc + 2 * (get_1d_idx(half_Lx, Ly, Lz) * m_input * Ns * n_color);
    }
    template <typename Float>
    QCU_DEVICE Float* getGatheredColorSpinorAddr (  Float* base_pc, qcu::QcuLattDesc &half_desc,
                                                    int n_color, int m_input) const
    {
        return getGatheredColorSpinorAddr(base_pc, half_desc.X(), half_desc.Y(), half_desc.Z(), half_desc.T(), n_color, m_input);
    }

    QCU_DEVICE int get_1d_idx (int half_Lx, int Ly, int Lz) const {
        return IDX4D(T(), Z(), Y(), X(), Lz, Ly, half_Lx);
    }

    QCU_DEVICE int X() const { return dims[X_DIM]; }
    QCU_DEVICE int Y() const { return dims[Y_DIM]; }
    QCU_DEVICE int Z() const { return dims[Z_DIM]; }
    QCU_DEVICE int T() const { return dims[T_DIM]; }
    QCU_DEVICE int Parity() const { return parity_; }

private:
    int dims[Nd];
    int parity_;
};
