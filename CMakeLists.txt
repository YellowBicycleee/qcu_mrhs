cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 89)    # for 4060 laptop
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -G")
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0") # -G 表示生成GPU调试信息，-O0 禁止优化以便于调试
# set(CMAKE_BUILD_TYPE Debug) # 设定默认构建类型为Debug

project(QCU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}

# shared library
add_library(qcu 
    SHARED
    ${PROJECT_SOURCE_DIR}/src/interface.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_interface.cu  
    ${PROJECT_SOURCE_DIR}/src/qcd/qcu_wilson_dslash.cu
    ${PROJECT_SOURCE_DIR}/src/data_format/qcu_data_format_shift.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_complex_axpby.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_complex_axpbypcz.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_complex_dotc.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_complex_norm.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_complex_xsay.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_complex_xpay.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_elementwise_div.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_elementwise_init.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_elementwise_mul.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_transpose_2d.cu
    ${PROJECT_SOURCE_DIR}/src/solver/bicgstab.cu
    ${PROJECT_SOURCE_DIR}/src/solver/bicgstab_solve.cu
	${PROJECT_SOURCE_DIR}/src/solver/bicgstab_interface.cu
)
target_include_directories(qcu 
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    /usr/local/cuda/bin/../targets/x86_64-linux/include
)
target_link_libraries(qcu
	cudart
	cudadevrt
    cublas
)

add_subdirectory(tests)
