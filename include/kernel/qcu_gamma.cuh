//
// Created by wangj on 2024/11/13.
//

#pragma once
#include "cuda_utils.cuh"

namespace qcu::kernel {

// calculate 1 + gamma, if dagger, just set col(1) = -col(1)
// for example,
// ---------------------------------------
// 1 + gamma_1 =
//      [ 1  0  0  i]
//      [ 0  1  i  0]
//      [ 0 -i  1  0]
//      [-i  0  0  1]
// ---------------------------------------
template <typename FloatType>
class Gamma1 {
protected:
    static QCU_DEVICE Complex<FloatType> get_elem (int row, int col) {
        if (row == 0) {
            if (col == 0) return {1, 0};
            else if (col == 3) return {0, 1};
            else if (col == 1 || col == 2) return {0, 0};
        }
        else if (row == 1) {
            if (col == 1) return {1, 0};
            else if (col == 2) return {0, 1};
            else if (col == 0 || col == 3) return {0, 0};
        }
        else if (row == 2) {
            if (col < 4) {
                return Complex<FloatType>(0, -1) * get_elem(1, col);
            }
        }
        else if (row == 3) {
            if (col < 4) {
                return Complex<FloatType>(0, -1) * get_elem(0, col);
            }
        }
        printf("Fatal: gamma_1[%d, %d] is not exist\n", row + 1, col + 1);
        cuda_abort();
        return {0, 0};
    }
    static QCU_DEVICE int get_reconstruct_mat_id (int row) {
        if (row == 0 || row == 1) return (3 - row);
        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return -1;
    }
    static QCU_DEVICE Complex<FloatType> get_reconstruct_scale (int row) {
        if (row == 0 || row == 1) return {0, -1};

        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return {0, 0};
    }
    // only row 0 and 1 have projection scale
    static QCU_DEVICE Complex<FloatType> get_projection_scale (int row) {
        if (row == 0 || row == 1) return {0, 1};

        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return {0, 0};
    }
};

// ---------------------------------------
// 1 + gamma_2 =
//      [ 1  0  0 -1]
//      [ 0  1  1  0]
//      [ 0  1  1  0]
//      [-1  0  0  1]
// ---------------------------------------
template <typename FloatType>
class Gamma2 {
protected:
    static QCU_DEVICE Complex<FloatType> get_elem (int row, int col) {
        if (row == 0) {
            if (col == 0) return {1, 0};
            else if (col == 3) return  {0, -1};
            else if (col == 1 || col == 2) return {0, 0};
        }
        else if (row == 1) {
            if (col == 1 || col == 2) return {1, 0};
            else if (col == 0 || col == 3) return {0, 0};
        }
        else if (row == 2 || row == 3) {
            if (col < 4) {
                if (row == 2) return get_elem(1, col);
                else return Complex<FloatType>{-1, 0} * get_elem(1, col);
            }
        }

        printf("Fatal: gamma_2[%d, %d] is not exist\n", row + 1, col + 1);
        cuda_abort();
        return {0, 0};
    }
    static QCU_DEVICE int get_reconstruct_mat_id (int row) {
        if (row == 0 || row == 1) return (3 - row);

        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return -1;
    }
    static QCU_DEVICE Complex<FloatType> get_reconstruct_scale (int row) {
        if (row == 0) return {-1, 0};
        if (row == 1) return {1, 0};

        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return {0, 0};
    }
    static QCU_DEVICE Complex<FloatType> get_projection_scale (int row) {
        if (row == 0) return {-1, 0};
        else if (row == 1) return {1, 0};

        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return {0, 0};
    }
};

// ---------------------------------------
// 1 + gamma_3 =
//      [ 1  0  i   0]
//      [ 0  1  0  -i]
//      [-i  0  1   0]
//      [ 0  i  0   1]
// ---------------------------------------
template <typename FloatType>
class Gamma3 {
protected:
    static QCU_DEVICE Complex<FloatType> get_elem (int row, int col) {
        if (row == 0) {
            if (col == 0) return {1, 0};
            else if (col == 2) return {0, 1};
            else if (col == 1 || col == 3) return {0, 0};
        }
        else if (row == 1) {
            if (col == 1) return {1, 0};
            else if (col == 3) return {0, -1};
            else if (col == 0 || col == 2) return {0, 0};
        }
        else if (row == 2) {
            if (col < 4) { return Complex<FloatType>{0, -1} * get_elem(0, col); }
        }
        else if (row == 3) {
            if (col < 4) {
                return Complex<FloatType>{0, 1} * get_elem(1, col);
            }
        }
        printf("Fatal: gamma_3[%d, x] is not exist\n", row + 1);
        cuda_abort();
        return {0, 0};
    }
    static QCU_DEVICE int get_reconstruct_mat_id (int row) {
        if (row == 0 || row == 1) return (2 + row);
        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return -1;
    }
    static QCU_DEVICE Complex<FloatType> get_reconstruct_scale (int row) {
        if (row == 0) return {0, -1};
        else if (row == 1) return {0, 1};

        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return {0, 0};
    }
    static QCU_DEVICE Complex<FloatType> get_projection_scale (int row) {
        if (row == 0) return {0, 1};
        else if (row == 1) return {0, -1};

        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return {0, 0};
    }
};



// ---------------------------------------
// 1 + gamma_4 =
//      [ 1  0  1  0]
//      [ 0  1  0  1]
//      [ 1  0  1  0]
//      [ 0  1  0  1]
// ---------------------------------------
template <typename FloatType>
class Gamma4 {
protected:
    static QCU_DEVICE Complex<FloatType> get_elem (int row, int col) {
        if (row == 0 || row == 2) {
            if (col == 0 || col == 2) return {1, 0};
            else if (col == 1 || col == 3) return {0, 0};
        }
        else if (row == 1 || row == 3) {
            if (col == 1 || col == 3) return {1, 0};
            else if (col == 0 || col == 2) return {0, 0};
        }
        printf("Fatal: gamma_4[%d, %d] is not exist\n", row + 1, col + 1);
        cuda_abort();
        return {0, 0};
    }
    static QCU_DEVICE int get_reconstruct_mat_id (int row) {
        if (row == 0 || row == 1) return (2 + row);

        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return -1;
    }
    static QCU_DEVICE Complex<FloatType> get_reconstruct_scale (int row) {
        if (row == 0 || row == 1) return {1, 0};
        printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        cuda_abort();
        return {0, 0};
    }
    static QCU_DEVICE Complex<FloatType> get_projection_scale (int row) {
        if (row == 0 || row == 1) return {1, 0};
        else {
            printf("Fatal: file %s line %d, row must be 0 or 1, but it is %d\n", __FILE__, __LINE__, row);
        }
        cuda_abort();
        return {0, 0};
    }
};

template <typename FloatType>
class Gamma : public Gamma1<FloatType>, public Gamma2<FloatType>, public Gamma3<FloatType>, public Gamma4<FloatType> {
public:
    static QCU_DEVICE Complex<FloatType> get_elem (int gamma_id, int row, int col) {
        switch (gamma_id) {
            case X_DIM: {
                return Gamma1<FloatType>::get_elem(row, col);
            } break;
            case Y_DIM: {
                return Gamma2<FloatType>::get_elem(row, col);
            } break;
            case Z_DIM: {
                return Gamma3<FloatType>::get_elem(row, col);
            } break;
            case T_DIM: {
                return Gamma4<FloatType>::get_elem(row, col);
            } break;
            default: {
                printf("Fatal: Wrong gamma_id\n");
                cuda_abort();
                return {0, 0};
            } break;
        }
    }

    static QCU_DEVICE int get_reconstruct_mat_id (int gamma_id, int row) {
        switch (gamma_id) {
            case X_DIM: {
                return Gamma1<FloatType>::get_reconstruct_mat_id(row);
            } break;
            case Y_DIM: {
                return Gamma2<FloatType>::get_reconstruct_mat_id(row);
            } break;
            case Z_DIM: {
                return Gamma3<FloatType>::get_reconstruct_mat_id(row);
            } break;
            case T_DIM: {
                return Gamma4<FloatType>::get_reconstruct_mat_id(row);
            } break;
            default: {
                printf("Fatal: Wrong gamma_id\n");
                cuda_abort();
                return -1;
            } break;
        }
    }

    static QCU_DEVICE Complex<FloatType> get_reconstruct_scale (int gamma_id, int row) {
        switch (gamma_id) {
            case X_DIM: {
                return Gamma1<FloatType>::get_reconstruct_scale(row);
            } break;
            case Y_DIM: {
                return Gamma2<FloatType>::get_reconstruct_scale(row);
            } break;
            case Z_DIM: {
                return Gamma3<FloatType>::get_reconstruct_scale(row);
            } break;
            case T_DIM: {
                return Gamma4<FloatType>::get_reconstruct_scale(row);
            } break;
            default: {
                printf("Fatal: Wrong gamma_id\n");
                cuda_abort();
                return {0, 0};
            } break;
        }
    }

    static QCU_DEVICE Complex<FloatType> get_projection_scale (int gamma_id, int row) {
        switch (gamma_id) {
            case X_DIM: {
                return Gamma1<FloatType>::get_projection_scale(row);
            }
            case Y_DIM: {
                return Gamma2<FloatType>::get_projection_scale(row);
            }
            case Z_DIM: {
                return Gamma3<FloatType>::get_projection_scale(row);
            }
            case T_DIM: {
                return Gamma4<FloatType>::get_projection_scale(row);
            }
            default: {
                printf("Fatal: Wrong gamma_id\n");
                cuda_abort();
            }
            break;;
        }
        cuda_abort();
        return {0, 0};
    }
};

// calculate 1 + gamma, if dagger, just set col(1) = -col(1)
// for example,
// ---------------------------------------
// 1 + gamma_1 =
//      [ 1  0  0  i]
//      [ 0  1  i  0]
//      [ 0 -i  1  0]
//      [-i  0  0  1]
// ---------------------------------------
// we can see that (1 + gamma_1) row(2, 3) = row(0, 1) * (-i, -i), so we constrain row to 0, 1
// only 2 columns have elem, so we constrain col to 0, 1

// template <typename _FloatType>
// QCU_DEVICE
// Complex<_FloatType> get_scale(int gamma_id, int row) {
//     kernel::Gamma<_FloatType> gamma;
//     if (gamma_id < 0 || gamma_id > 3) {
//         printf("Fatal: gamma_id %d out of range\n", gamma_id + 1);
//         assert(0);
//     }
//     if (!(row == 0 || row == 1)) {
//         printf("Fatal: row or col out of range\n");
//         assert(0);
//     }
//     // 1 + gamma_1
//     if (gamma_id == 0 || gamma_id == 1) {
//         if (row == 0) return gamma.get_elem(gamma_id, 0, 3);
//         else return gamma.get_elem(gamma_id, 1, 2);
//     }
//
//     else if (gamma_id == 2 || gamma_id == 3) {
//         if (row == 0) return gamma.get_elem(gamma_id, 0, 2);
//         else return gamma.get_elem(gamma_id, 1, 3);
//     }
//
//     // error handling
//     printf("gamma_id = %d, row = %d,some parameter out of range\n", gamma_id, row);
//     cuda_abort();
//     return {0, 0};
// }

}

