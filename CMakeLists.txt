cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 80)    # for A100
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

project(QCU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}

# shared library
add_library(qcu 
    SHARED
    ${PROJECT_SOURCE_DIR}/src/interface.cpp
    ${PROJECT_SOURCE_DIR}/src/qcu_interface.cpp  
    ${PROJECT_SOURCE_DIR}/src/qcd/qcu_wilson_dslash.cu
    ${PROJECT_SOURCE_DIR}/src/data_format/qcu_data_format_shift.cu
)
target_include_directories(qcu 
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
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
# # static library
# add_library(qcu_static
#     STATIC 
#     ${PROJECT_SOURCE_DIR}/src/interface.cpp
#     ${PROJECT_SOURCE_DIR}/src/qcu_interface.cpp   
# )

# target_link_libraries(qcu_static
# 	cudart
# 	cudadevrt
# )

# target_include_directories(qcu_static 
#     PUBLIC
#     ${PROJECT_SOURCE_DIR}/include
# )
