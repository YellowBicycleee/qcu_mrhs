//
// Created by wangj on 2024/11/13.
//

#pragma once

namespace qcu {

namespace kernel {

// calculate 1 + gamma, if dagger, just set col(1) = -col(1)
// for example,
// ---------------------------------------
// 1 + gamma_1 =
//      [ 1  0  0  i]
//      [ 0  1  i  0]
//      [ 0 -i  1  0]
//      [-i  0  0  1]
// ---------------------------------------
// 1 + gamma_2 =
//      [ 1  0  0 -1]
//      [ 0  1  1  0]
//      [ 0  1  1  0]
//      [-1  0  0  1]
// ---------------------------------------
// 1 + gamma_3 =
//      [ 1  0  i   0]
//      [ 0  1  0  -i]
//      [-i  0  1   0]
//      [ 0  i  0   1]
// ---------------------------------------
// 1 + gamma_4 =
//      [ 1  0  1  0]
//      [ 0  1  0  1]
//      [ 1  0  1  0]
//      [ 0  1  0  1]
// ---------------------------------------

template <typename _FloatType>
class Gamma1 {
protected:
    QCU_DEVICE Complex<_FloatType> get_elem (int row, int col) {
        if (row == 0) {
            if (col == 0) return Complex<_FloatType>(1, 0);
            else if (col == 3) return Complex<_FloatType>(0, 1);
            else if (col == 1 || col == 2) return Complex<_FloatType>(0, 0);
            else {
                printf("Fatal: gamma_1[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
        }
        else if (row == 1) {
            if (col == 0) return Complex<_FloatType>(0, 0);
            else if (col == 1) return Complex<_FloatType>(1, 0);
            else if (col == 2) return Complex<_FloatType>(0, 1);
            else if (col == 3) return Complex<_FloatType>(0, 0);
            else {
                printf("Fatal: gamma_1[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
        }
        else if (row == 2) {
            if (col > 3) {
                printf("Fatal: gamma_1[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
            return Complex<_FloatType>(0, -1) * get_elem(1, col);
        }
        else if (row == 3) {
            if (col > 3) {
                printf("Fatal: gamma_1[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
            return Complex<_FloatType>(0, -1) * get_elem(0, col);
        }
        else {
            printf("Fatal: gamma_1[%d, %d] is not exist\n", row + 1, col + 1);
            errorQcu("Fatal Error\n");
        }
    }
};

template <typename _FloatType>
class Gamma2 {
protected:
    QCU_DEVICE Complex<_FloatType> get_elem (int row, int col) {
        if (row == 0) {
            if (col == 0) return Complex<_FloatType>(1, 0);
            else if (col == 3) return  Complex<_FloatType>(0, -1);
            else if (col == 1 || col == 2) return Complex<_FloatType>(0, 0);
            else {
                printf("Fatal: gamma_2[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
        }
        else if (row == 1) {
            if (col == 1 || col == 2) return Complex<_FloatType>(1, 0);
            else if (col == 0 || col == 3) return Complex<_FloatType>(0, 0);
            else {
                printf("Fatal: gamma_2[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
        }
        else if (row == 2 || row == 3) {
            if (col > 3) {
                printf("Fatal: gamma_2[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
            if (row == 2) return get_elem(1, col);
            else return Complex<_FloatType>(-1, 0) * get_elem(1, col);
        }
        else {
            printf("Fatal: gamma_2[%d, %d] is not exist\n", row + 1, col + 1);
            errorQcu("Fatal Error\n");
        }
    }
};

template <typename _FloatType>
class Gamma3 {
protected:
    QCU_DEVICE Complex<_FloatType> get_elem (int row, int col) {
        if (row == 0) {
            if (col == 0) return Complex<_FloatType>(1, 0);
            else if (col == 2) return Complex<_FloatType>(0, 1);
            else if (col == 1 || col == 3) return Complex<_FloatType>(0, 0);
            else {
                printf("Fatal: gamma_3[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
        }
        else if (row == 1) {
            if (col == 1) return Complex<_FloatType>(1, 0);
            else if (col == 3) return Complex<_FloatType>(0, -1);
            else if (col == 0 || col == 2) return Complex<_FloatType>(0, 0);
            else {
                printf("Fatal: gamma_3[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
        }
        else if (row == 2) {
            if (col > 3) {
                printf("Fatal: gamma_2[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
            return Complex<_FloatType>(0, -1) * get_elem(0, col);
        }
        else if (row == 3) {
            if (col > 3) {
                printf("Fatal: gamma_2[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
            return Complex<_FloatType>(0, 1) * get_elem(1, col);
        }
    }
};

template <typename _FloatType>
class Gamma4 {
protected:
    QCU_DEVICE Complex<_FloatType> get_elem (int row, int col) {
        if (row == 0 || row == 2) {
            if (col == 0 || col == 2) return Complex<_FloatType>(1, 0);
            else if (col == 1 || col == 3) return Complex<_FloatType>(0, 0);
            else {
                printf("Fatal: gamma_4[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
        }
        else if (row == 1 || row == 3) {
            if (col == 1 || col == 3) return Complex<_FloatType>(1, 0);
            else if (col == 0 || col == 2) return Complex<_FloatType>(0, 0);
            else {
                printf("Fatal: gamma_4[%d, %d] is not exist\n", row + 1, col + 1);
                errorQcu("Fatal Error\n");
            }
        }
        else {
            printf("Fatal: gamma_4[%d, %d] is not exist\n", row + 1, col + 1);
            errorQcu("Fatal Error\n");
        }
    }
};

template <typename _FloatType>
class Gamma : public Gamma1<_FloatType>, public Gamma2<_FloatType>, public Gamma4<_FloatType> {
    QCU_DEVICE Complex<_FloatType> get_elem (int gamma_id, int row, int col) {
        switch (gamma_id) {
            case X_DIM: {
                return Gamma1<_FloatType>::get_elem_x(row, col);
            } break;
            case Y_DIM: {
                return Gamma2<_FloatType>::get_elem_y(row, col);
            } break;
            case Z_DIM: {
                return Gamma3<_FloatType>::get_elem_z(row, col);
            } break;
            case T_DIM: {
                return Gamma4<_FloatType>::get_elem(row, col);
            } break;
            default: {
                errorQcu("Wrong gamma_id\n");
            } break;
        }
    }
};

}

}