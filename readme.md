# QCU_MRHS

## 1 Introduction

This is a repo with `SU(N)` dslash BiCGStab Solver.

## 2. Supported Function

- SU(N) MRHS Wilson Dslash
  - Now supported  Nvidia GPUS with compute capablity 8.x (TensorOp Used)
  - SIMT for general GPUs
  - Using cuda12 and above is safe, cuda under 11.6 may result in compiling failure. 
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

### 3.3 MPI required
If your environment supports CUDA-aware MPI, you can use it by changing `src/qcu_config/qcu_config.cu` 
set cuda_aware_mpi_supported_flag = true.

If not, you can use `src/qcu_config/qcu_config.cu` set cuda_aware_mpi_supported_flag = false.
Then you can use `mpirun` to run your program.

   
## 4. Developing
    At the beginning of developing, I used `clang-format`. But recently,
I decided to stop using `clang-format`, referencing to `cutlass`.
