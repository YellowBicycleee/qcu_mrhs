#pragma once

#include "qcu_helper.h"

namespace qcu {

// device level api
namespace device {

namespace solver {

/// @brief Precondition for solver
/// @tparam _Tp_Input 
/// @tparam _Tp_Output 
template <
    /// output type, such as complex<double>, complex<float> complex<half>
    typename _Tp_Output,
    //  input type
    typename _Tp_Input
>
class SolverPreconditioner {

public: 
    // Arguments
    struct Arguments {
        _Tp_Output* output;
        _Tp_Input*  input;

        QCU_HOST_DEVICE
        Arguments() : output(nullptr), input(nullptr) {}

        QCU_HOST_DEVICE
        Arguments(_Tp_Output* output_, _Tp_Input* input) 
            : output(output_), input(input_)
        {}
    };
public:
    // default constructor
    SolverPreconditioner () = default;
    QcuStatus run (cudaStream_t stream = nullptr) {
        throw int;  // TODO
    }

    // Runs the kernel using initialized state.
    QcuStatus operator() (cudaStream_t stream = nullptr) {
        return this->run(stream);
    }
};

/// @brief 
/// @tparam _Tp 迭代类型
template <typename _Tp>
class Solver {

public:
    struct Arguments {
        /* data */
        QCU_HOST_DEVICE
        Arguments () {}

        // constructor with arguments
    };
    
};

}

}

}