//
// Created by wjc on 24-10-22.
//

#pragma once
#include "qcu_helper.h"
enum DslashType {
    kDslashWilson = 0,
    kDslashStaggered = 1,
    kDslashUnkown
};

enum QcuNspin {
    kNspinWilson = 4,
    kNspinStaggered = 1,
    kNspinUndefined
};

enum QcuStaggeredPhase {
    kQcuStaggeredPhaseNo = 0,
    kQcuStaggeredPhaseMilc = 1,
    kQcuStaggeredPhaseCps = 2,
    kQcuStaggeredPhaseTifr = 3,
    kQcuStaggeredPhaseNoInvalid
};

enum QcuDaggerFlag {
    kDaggerNo = 0,
    kDaggerYes,
    kDaggerUndefined
};