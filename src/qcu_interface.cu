#include <cassert>

#include "qcu_interface.h"
#include "qcu_macro.h"
// clang-format off

namespace qcu {

int gpu_id = -1;    // not initialized

// TODO
void Qcu::getDslash (DSLASH_TYPE dslashType, double mass, int nColors, int nInputs, QCU_PRECISION floatPrecision, int daggerFlag = 0) {
    errorQcu("Not implemented yet\n");  // TODO
}
// TODO
void Qcu::startDslash (int parity, int daggerFlag = 0) {
    errorQcu("Not implemented yet\n");  // TODO
}
// TODO
void Qcu::loadGauge (void *gauge) {
    errorQcu("Not implemented yet\n");  // TODO
}
}  // namespace qcu