# QCU_MRHS

## 1 Introduction

This is a repo with `SU(N)` dslash BiCGStab Solver.

## 2. Supported Function

- SU(N) Dslash
  - Now supported  Nvidia GPUS with compute capablity 8.x (TensorOp Used)
  - **TODO:** SIMT version
- BiCGStab Based on SU(N) dslash

## 3. Compile
### 3.1 **Other Repos Required**

  1. qcu_io: 
      ```SHELL
      git clone https://github.com/YellowBicycleee/qcu_io.git
      ```
  2. PyQuda
      ```SHELL
     git clone https://github.com/YellowBicycleee/PyQuda.git
     ```
     
### 3.2 **QCU_MRHS**

Then compile qcu_mrhs. Assume you are in `qcu_mrhs` directory.

1. compile qcu.
    ```SHELL
    mkdir build 
    cd build 
    cmake .. -DCMAKE_CUDA_COMPILER=xxx # (xxx is your path)
    make -j 12
    ```