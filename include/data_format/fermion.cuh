#pragma once

#include <array>
#include <complex/qcu_complex.cuh>
#include <vector>

#include "check_error/check_cuda.cuh"
#include "desc/qcu_desc.h"

namespace qcu {

// 异常不安全版本，可能暂时没时间去做一个异常安全版本了。。。
template <
    int Ndim_ = Nd,
    int Nspin_ = 4
>
struct FermionGhost {
    FermionGhost(
        qcu::QcuLattDesc const& latt_desc_local,
        unsigned int multiprogress_mask,
        int n_color, int m_rhs, QcuPrecision precision)
    {   // lattice_volume() / 2 : even odd precondition
        // Nspin_ / 2 : projection
        size_t total_length = latt_desc_local.lattice_volume() / 2 * Nspin_ / 2 * n_color * m_rhs;
        size_t size_complex = 0;
        switch (precision) {
            case QcuPrecision::kPrecisionHalf:
                size_complex = sizeof(qcu::Complex<half>);
                break;
            case QcuPrecision::kPrecisionSingle:
                size_complex = sizeof(qcu::Complex<float>);
                break;
            case QcuPrecision::kPrecisionDouble:
                size_complex = sizeof(qcu::Complex<double>);
                break;
            default:
                errorQcu("Undefined precision\n");
        }

        for (int i = 0; i < Ndim_; ++i) {
            if (multiprogress_mask & (1 << i)) {
                void* ptr_pack;
                void* ptr_unpack;

                void* ptr_host_pack;
                void* ptr_host_unpack;
                // forwards
                CHECK_CUDA(cudaMalloc(&ptr_pack, total_length / latt_desc_local.at(i) * size_complex));
                CHECK_CUDA(cudaMalloc(&ptr_unpack, total_length / latt_desc_local.at(i) * size_complex));
                ghost_pack_cell.push_back(ptr_pack);
                ghost_unpack_cell.push_back(ptr_unpack);
                // backwards
                CHECK_CUDA(cudaMalloc(&ptr_pack, total_length / latt_desc_local.at(i) * size_complex));
                CHECK_CUDA(cudaMalloc(&ptr_unpack, total_length / latt_desc_local.at(i) * size_complex));
                ghost_pack_cell.push_back(ptr_pack);
                ghost_unpack_cell.push_back(ptr_unpack);

                // host forwards
                CHECK_CUDA(cudaMallocHost(&ptr_host_pack, total_length / latt_desc_local.at(i) * size_complex));
                CHECK_CUDA(cudaMallocHost(&ptr_host_unpack, total_length / latt_desc_local.at(i) * size_complex));
                host_ghost_pack_cell.push_back(ptr_host_pack);
                host_ghost_unpack_cell.push_back(ptr_host_unpack);
                // host backwards
                CHECK_CUDA(cudaMallocHost(&ptr_host_pack, total_length / latt_desc_local.at(i) * size_complex));
                CHECK_CUDA(cudaMallocHost(&ptr_host_unpack, total_length / latt_desc_local.at(i) * size_complex));
                host_ghost_pack_cell.push_back(ptr_host_pack);
                host_ghost_unpack_cell.push_back(ptr_host_unpack);

                ghost_len[i] = total_length / latt_desc_local.at(i);
            } else {
                // forward and backward
                ghost_pack_cell.push_back(nullptr);
                ghost_pack_cell.push_back(nullptr);
                ghost_unpack_cell.push_back(nullptr);
                ghost_unpack_cell.push_back(nullptr);
                // host forward and backward
                host_ghost_pack_cell.push_back(nullptr);
                host_ghost_pack_cell.push_back(nullptr);
                host_ghost_unpack_cell.push_back(nullptr);
                host_ghost_unpack_cell.push_back(nullptr);
                ghost_len[i] = 0;
            }
        }
        assert(Ndim_ * 2 == ghost_pack_cell.size() && Ndim_ * 2 == ghost_unpack_cell.size());
    }

    FermionGhost (const FermionGhost&) = delete;
    FermionGhost& operator=(const FermionGhost&) = delete;

    ~FermionGhost() {

        for (int i = 0; i < Ndim_ * 2; ++i) {
            if (ghost_pack_cell[i] != nullptr) {
                CHECK_CUDA(cudaFree(ghost_pack_cell[i]));
            }
            if (ghost_unpack_cell[i] != nullptr) {
                CHECK_CUDA(cudaFree(ghost_unpack_cell[i]));
            }
            if (host_ghost_pack_cell[i] != nullptr) {
                CHECK_CUDA(cudaFreeHost(host_ghost_pack_cell[i]));
            }
            if (host_ghost_unpack_cell[i] != nullptr) {
                CHECK_CUDA(cudaFreeHost(host_ghost_unpack_cell[i]));
            }

        }
    }

    void* get_pack_buf_at(int dim, int dir) {
        if (dim >= 0 && dim <= Ndim_ && (dir == BWD || dir == FWD)) {
            return ghost_pack_cell[dim * 2 + dir];
        }
        else {
            throw std::runtime_error("Dim out of range");
        }
    }
    void* get_unpack_buf_at(int dim, int dir) {
        if (dim >= 0 && dim <= Ndim_ && (dir == BWD || dir == FWD)) {
            return ghost_unpack_cell[dim * 2 + dir];
        }
        else {
            throw std::runtime_error("Dim out of range");
        }
    }

    void* get_host_pack_buf_at(int dim, int dir) {
        if (dim >= 0 && dim <= Ndim_) {
            return host_ghost_pack_cell[dim * 2 + dir];
        }
        else {
            throw std::runtime_error("Dim out of range");
        }
    }
    void* get_host_unpack_buf_at(int dim, int dir) {
        if (dim >= 0 && dim <= Ndim_) {
            return host_ghost_unpack_cell[dim * 2 + dir];
        }
        else {
            throw std::runtime_error("Dim out of range");
        }
    }

    size_t complex_buff_len = 0;
    std::vector<void*> ghost_pack_cell;
    std::vector<void*> ghost_unpack_cell;
    std::vector<void*> host_ghost_pack_cell;
    std::vector<void*> host_ghost_unpack_cell;
    std::array<size_t, Nd> ghost_len;
};

template<
    int Nspin_ = 4
>
struct Fermion {
    Fermion(qcu::QcuLattDesc const& latt_desc_local,
        unsigned int multiprogress_mask,
        int n_color, int m_rhs, QcuPrecision precision)
    : complex_buff_len(latt_desc_local.lattice_volume() * Nspin_ * n_color * m_rhs)
    {
        size_t size_complex = 0;
        switch (precision) {
            case QcuPrecision::kPrecisionHalf:
                size_complex = sizeof(qcu::Complex<half>);
            break;
            case QcuPrecision::kPrecisionSingle:
                size_complex = sizeof(qcu::Complex<float>);
            break;
            case QcuPrecision::kPrecisionDouble:
                size_complex = sizeof(qcu::Complex<double>);
            break;
            default:
                errorQcu("Undefined precision\n");
        }
        CHECK_CUDA(cudaMalloc(&fermion, complex_buff_len * size_complex));
    }

    Fermion (const Fermion&) = delete;
    Fermion& operator=(const Fermion&) = delete;

    ~Fermion() {
        CHECK_CUDA(cudaFree(fermion));
    }

    void* get_fermion() {
        return fermion;
    }

    size_t complex_buff_len = 0;
    void* fermion;
};

}