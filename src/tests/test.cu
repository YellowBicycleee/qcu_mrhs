#include "qcd/qcu_dslash.h"
#include "qcu_macro.h"

using namespace qcu;
int main () {
    QCU_PRECISION prec = QCU_DOUBLE_PRECISION;
    int Lx = 4, Ly = 4, Lz = 4, Lt = 4;
    int nColor = 3;
    int mInput = 1;
    double kappa = 1.0;
    int parity = 0;
    bool daggerFlag = false;
    void* fermionIn_MRHS;
    void* fermionOut_MRHS;
    void* gauge;
    QcuLattDesc lattDesc(Lx, Ly, Lz, Lt);
    QcuProcDesc procDesc(1, 1, 1, 1);

    int vol = Lx * Ly * Lz * Lt;
    int colorSpinor_vlen = vol * 4 * nColor;
    int gauge_vlen = 4 * vol * nColor * nColor;
    CHECK_CUDA(cudaMalloc(&fermionIn_MRHS, sizeof(double) * colorSpinor_vlen * 2 / 2));
    CHECK_CUDA(cudaMalloc(&fermionOut_MRHS, sizeof(double) * colorSpinor_vlen * 2 / 2));
    CHECK_CUDA(cudaMalloc(&gauge, sizeof(double) * gauge_vlen * 2));

    DslashParam param(prec, nColor, mInput, kappa, parity, daggerFlag, fermionIn_MRHS, 
                        fermionOut_MRHS, gauge, lattDesc, procDesc);

    WilsonDslash dslash(param);
    dslash.apply();

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(fermionIn_MRHS));
    CHECK_CUDA(cudaFree(fermionOut_MRHS));
    CHECK_CUDA(cudaFree(gauge));

}