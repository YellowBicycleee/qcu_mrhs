#pragma once

#include "qcu_helper_macro.h"
namespace qcu::device {
// FIXME need to check this with odd local volumes
// template <int dim, typename Arg>
template <typename Real_>
__device__ __forceinline__
constexpr Real_ get_phase(int dim, int x, int y, int z, int t, QcuStaggeredPhase staggered_phase, int Lt, Real_ t_boundary) {

    Real_ phase = Real_(1.0);

    if (staggered_phase == kQcuStaggeredPhaseMilc) {
        if (dim == 0) {
            phase = (1.0 - 2.0 * (t % 2) );
        } else if (dim == 1) {
            phase = (1.0 - 2.0 * ((t + x) % 2) );
        } else if (dim == 2) {
            phase = (1.0 - 2.0 * ((t + x + y) % 2) );
        } else if (dim == 3) { // also apply boundary condition
            phase = (t == Lt-1) ? t_boundary : Real_(1.0);
        }
    } else if (staggered_phase == kQcuStaggeredPhaseTifr) {
        if (dim==0) {
	        phase = (1.0 - 2.0 * ((3 + t + z + y) % 2) );
        } else if (dim == 1) {
	        phase = (1.0 - 2.0 * ((2 + t + z) % 2) );
        } else if (dim == 2) {
	        phase = (1.0 - 2.0 * ((1 + t) % 2) );
        } else if (dim == 3) { // also apply boundary condition
            phase = (t == Lt - 1) ? t_boundary : Real_(1.0);
        }
    } else if (staggered_phase == kQcuStaggeredPhaseCps) {
        if (dim==0) {
            phase = 1.0;
        } else if (dim == 1) {
	        phase = (1.0 - 2.0 * ((1 + x) % 2) );
        } else if (dim == 2) {
	        phase = (1.0 - 2.0 * ((1 + x + y) % 2) );
        } else if (dim == 3) { // also apply boundary condition
            phase = ((t == Lt - 1) ? t_boundary : Real_(1.0)) * Real_(1.0 - 2 * ((1 + x + y + z) % 2) );
        }
    }
    return phase;
}

}