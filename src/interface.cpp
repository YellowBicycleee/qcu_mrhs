#include <cassert>
#include <cstdio>

#include "qcu.h"
static inline void qcuException(const char *message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(1);
}
static inline void notImplemented() { qcuException("Not implemented"); }

void initGridSize(QcuGrid *grid, QcuParam *p_param) { notImplemented(); }
void pushBackFermions(void *fermionOut, void *fermionIn) { notImplemented(); }
void qcuDslash(int parity, int daggerFlag) { notImplemented(); }
void qcuDslashXpay(int parity, int daggerFlag) { notImplemented(); }
void loadQcuGauge(void *gauge) { notImplemented(); }
void getDslash(int dslashType, double mass, int nColors, int nInputs, int floatPrecision, int mInput) {
    notImplemented();
}
void qcuInvert(void *x_vector, void *b_vector, void *gauge, QcuParam *param, double p_max_prec, double p_kappa) {
    notImplemented();
}
void finalizeQcu() { notImplemented(); }
