#include <vector>

#include "qcd/qcu_dslash.h"
#include "qcu.h"
#include "qcu_enum.h"
#include "qcu_macro.h"
using namespace qcu;

QcuGrid initGridSize(int Nx, int Ny, int Nz, int Nt) {
    QcuGrid grid;
    grid.grid_size[X_DIM] = Nx;
    grid.grid_size[Y_DIM] = Ny;
    grid.grid_size[Z_DIM] = Nz;
    grid.grid_size[T_DIM] = Nt;
    return grid;
}
QcuParam initQcuParam(int Lx, int Ly, int Lz, int Lt) {
    QcuParam param;
    param.lattice_size[X_DIM] = Lx;
    param.lattice_size[Y_DIM] = Ly;
    param.lattice_size[Z_DIM] = Lz;
    param.lattice_size[T_DIM] = Lt;
    return param;
}

void allocateFermion(vector<void*>& fermionArr, int mInput, int colorSpinor_vlen) {
    // memory alloaction
    void* fermion;
    for (int i = 0; i < mInput; ++i) {
        CHECK_CUDA(cudaMalloc(&fermion, 2 * sizeof(double) * colorSpinor_vlen));
        fermionArr.push_back(fermionIn);
    }
}

void freeFermion(vector<void*>& fermionArr, int mInput) {
    // TODO： 查一下pop_back
    void* fermion;
    for (int i = 0; i < mInput; i++) {
        fermion = fermionArr[i];
        CHECK_CUDA(cudaFree(fermion));
    }
    fermionArr.clear();
}

int main() {
    // QCU_PRECISION prec = QCU_DOUBLE_PRECISION;
    int Lx = 16;
    int Ly = 16;
    int Lz = 16;
    int Lt = 16;

    int Nx = 1;
    int Ny = 1;
    int Nz = 1;
    int Nt = 1;

    int nColor = 3;
    int mInput = 1;
    double kappa = 1.0;
    bool daggerFlag = false;

    QcuGrid process_grid = initGridSize(Nx, Ny, Nz, Nt);
    QcuParam qcu_latt_param = initQcuParam(Lx, Ly, Lz, Lt);
    int inputFloatPrecision = QCU_DOUBLE_PRECISION;
    int dslashFloatPrecision = QCU_HALF_PRECISION;

    void* gauge;

    int vol = Lx * Ly * Lz * Lt;
    int colorSpinor_vlen = vol * 4 * nColor;
    int gauge_vlen = 4 * vol * nColor * nColor;
    CHECK_CUDA(cudaMalloc(&gauge, sizeof(double) * gauge_vlen * 2));

    vector<void*> fermionIn_arr;
    vector<void*> fermionOut_arr;
    allocateFermion(fermionIn_arr, mInput, colorSpinor_vlen);
    allocateFermion(fermionOut_arr, mInput, colorSpinor_vlen);

    // begin
    initGridSize(&process_grid, qcu_latt_param, nColor, mInput, inputFloatPrecision, dslashFloatPrecision);
    getDslash(DSLASH_WILSON, -3.5);
    loadQcuGauge(gauge, inputFloatPrecision);

    for (int parity = 0; parity < 2; ++parity) {
        for (int i = 0; i < mInput; i++) {
            void* fermionIn = fermionIn_arr[i];
            void* fermionOut = fermionOut_arr[i];
            void* output = static_cast<void*>(static_cast<double*>(fermionOut) + pairty * 2 * colorSpinor_vlen / 2);
            void* input = static_cast<void*>(static_cast<double*>(fermionIn) + (1 - pairty) * 2 * colorSpinor_vlen / 2);
            pushBackFermions(output, input);
        }
        start_dslash(parity, daggerFlag);
    }

    // end and finalize QCU
    finalizeQcu();

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(gauge));
}