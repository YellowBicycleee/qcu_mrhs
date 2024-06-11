#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QcuParam_s {
    int lattice_size[4];
} QcuParam;

typedef struct QcuGrid_t {
    int grid_size[4];
} QcuGrid;

void initGridSize(QcuGrid *grid, QcuParam *p_param);
void pushBackFermions(void *fermionOut, void *fermionIn);
void qcuDslash(int parity, int daggerFlag);
void qcuDslashXpay(int daggerFlag);
void loadQcuGauge(void *gauge);  // double precision
void getDslash(int dslashType, double mass, int nColors, int nInputs, int floatPrecision, int mInput);
void qcuInvert(void *x_vector, void *b_vector, void *gauge, QcuParam *param, double p_max_prec, double p_kappa);
void finalizeQcu();

#ifdef __cplusplus
}
#endif
