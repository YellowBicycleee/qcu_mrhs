#pragma once

// clang-format off
enum QCU_PRECISION {   // precision
    QCU_HALF_PRECISION = 0, 
    QCU_SINGLE_PRECISION = 1,
    QCU_DOUBLE_PRECISION,
    QCU_PRECISION_UNDEFINED
};

enum QCU_PRECONDITION { 
    QCU_NO_PRECONDITION = 0,
    QCU_EO_PC_4D 
};

enum QCU_PARITY {
    EVEN_PARITY = 0,
    ODD_PARITY = 1,
    PARITY = 2
};

enum DIMS {
    X_DIM = 0,
    Y_DIM,
    Z_DIM,
    T_DIM,
    Nd
};

enum DIRS {
    BWD = 0,
    FWD = 1,
    DIRECTIONS
};

enum QCU_DAGGER_FLAG { 
    QCU_DAGGER_NO = 0, 
    QCU_DAGGER_YES, 
    QCU_DAGGER_UNDEFINED 
};

enum DSLASH_TYPE {
    DSLASH_WILSON = 0, 
    DSLASH_CLOVER,
    DSLASH_UNKNOWN
};

enum MemoryStorage {
    NON_COALESCED = 0,
    COALESCED = 1,
};

enum ShiftDirection {
    TO_COALESCE = 0,
    TO_NON_COALESCE = 1,
};
