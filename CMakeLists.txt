cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 80)    # for A100
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0") # -G 表示生成GPU调试信息，-O0 禁止优化以便于调试
# set(CMAKE_BUILD_TYPE Debug) # 设定默认构建类型为Debug

project(QCU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}

# shared library
add_library(qcu 
    SHARED
    ${PROJECT_SOURCE_DIR}/src/interface.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_interface.cu  
    ${PROJECT_SOURCE_DIR}/src/qcd/qcu_wilson_dslash.cu
    ${PROJECT_SOURCE_DIR}/src/data_format/qcu_data_format_shift.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_blas/qcu_blas.cu
)
target_include_directories(qcu 
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    /usr/local/cuda/bin/../targets/x86_64-linux/include
)
target_link_libraries(qcu
	cudart
	cudadevrt
)

add_executable(test_shift_data 
    ${PROJECT_SOURCE_DIR}/src/tests/test_shift_datatype.cu
)
target_include_directories(test_shift_data
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_shift_data
    qcu
)


add_executable(test_dslash
    ${PROJECT_SOURCE_DIR}/src/tests/test_wilson_dslash.cu
)
target_include_directories(test_dslash
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_dslash
    qcu
    cudart
)


add_executable(test_gather
    ${PROJECT_SOURCE_DIR}/src/tests/test_gather.cu
)
target_include_directories(test_gather
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_gather
    qcu
    cudart
)

add_executable(test_inner_product
    ${PROJECT_SOURCE_DIR}/src/tests/test_inner_prod/test_inner_prod.cu
)
target_link_libraries(test_inner_product
    cudart
    cublas
)
