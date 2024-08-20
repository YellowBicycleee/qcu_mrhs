cmake_minimum_required(VERSION 3.8)
# include (../cmake/FindCUDA.cmake)

set(CMAKE_CUDA_ARCHITECTURES 86)    # for 3090
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

project(QCU_TESTS LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# message("${PROJECT_SOURCE_DIR}/src/qcu_blas")

add_executable       (test_shift_data ${PROJECT_SOURCE_DIR}/test_shift_datatype/test_shift_datatype.cu)
target_link_libraries(test_shift_data qcu cublas)

add_executable       (test_dslash ${PROJECT_SOURCE_DIR}/test_dslash/test_wilson_dslash.cu)
target_link_libraries(test_dslash qcu cudart cublas)


add_executable       (test_gather ${PROJECT_SOURCE_DIR}/test_gather/test_gather.cu)
target_link_libraries(test_gather qcu cudart cublas)

add_executable       (test_inner_product ${PROJECT_SOURCE_DIR}/test_inner_prod/test_inner_prod.cu)
target_link_libraries(test_inner_product cudart cublas)

add_executable       (test_inner_product2 ${PROJECT_SOURCE_DIR}/test_inner_prod/test_inner_prod2.cu)
target_link_libraries(test_inner_product2 cudart cublas qcu)

add_executable       (test_inner_product_myself ${PROJECT_SOURCE_DIR}/test_inner_prod/test_inner_prod_myself.cu)
target_link_libraries(test_inner_product_myself cudart cublas qcu)

# ========================
add_executable       (test_norm ${PROJECT_SOURCE_DIR}/test_norm/test_norm.cu)
target_link_libraries(test_norm qcu cudart cublas)
target_include_directories(test_norm PUBLIC ${PROJECT_SOURCE_DIR}/../src/qcu_blas)



# ========================
add_executable       (test_transpose ${PROJECT_SOURCE_DIR}/test_transpose/test_transpose.cu)
target_link_libraries(test_transpose qcu cudart)
