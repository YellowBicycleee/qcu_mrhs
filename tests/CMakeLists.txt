cmake_minimum_required(VERSION 3.8)
# include (../cmake/FindCUDA.cmake)

set(CMAKE_CUDA_ARCHITECTURES 86)    # for 3090
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

project(QCU_TESTS LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# message("${PROJECT_SOURCE_DIR}/src/qcu_blas")
add_executable       (function_paramter_test ${PROJECT_SOURCE_DIR}/small_test/test_device_function_demo.cu)
target_link_libraries(function_paramter_test qcu cublas)

add_executable       (test_shift_data ${PROJECT_SOURCE_DIR}/test_shift_datatype/test_shift_datatype.cu)
target_link_libraries(test_shift_data qcu cublas)

add_executable       (test_dslash ${PROJECT_SOURCE_DIR}/test_dslash/test_wilson_dslash.cu)
target_link_libraries(test_dslash qcu cublas)


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

# CAXPBY
add_executable       (test_caxpby ${PROJECT_SOURCE_DIR}/test_blas/test_caxpby.cu)
target_link_libraries(test_caxpby qcu cudart cublas)
target_include_directories(test_caxpby PUBLIC ${PROJECT_SOURCE_DIR})

# CAXPBYPCZ
add_executable       (test_caxpbypcz ${PROJECT_SOURCE_DIR}/test_blas/test_caxpbypcz.cu)
target_link_libraries(test_caxpbypcz qcu cudart cublas)
target_include_directories(test_caxpbypcz PUBLIC ${PROJECT_SOURCE_DIR})

# CXSAY
add_executable       (test_cxsay ${PROJECT_SOURCE_DIR}/test_blas/test_cxsay.cu)
target_link_libraries(test_cxsay qcu cudart cublas)
target_include_directories(test_cxsay PUBLIC ${PROJECT_SOURCE_DIR})

# CXPAY
add_executable       (test_cxpay ${PROJECT_SOURCE_DIR}/test_blas/test_cxpay.cu)
target_link_libraries(test_cxpay qcu cudart cublas)
target_include_directories(test_cxpay PUBLIC ${PROJECT_SOURCE_DIR})

# ELEMENWISE_DIV
add_executable       (test_elementwise_div ${PROJECT_SOURCE_DIR}/test_blas/test_elementwise_div.cu)
target_link_libraries(test_elementwise_div qcu cudart cublas)
target_include_directories(test_elementwise_div PUBLIC ${PROJECT_SOURCE_DIR})
# ELEMENWISE_MUL
add_executable       (test_elementwise_mul ${PROJECT_SOURCE_DIR}/test_blas/test_elementwise_mul.cu)
target_link_libraries(test_elementwise_mul qcu cudart cublas)
target_include_directories(test_elementwise_mul PUBLIC ${PROJECT_SOURCE_DIR})

# ELEMENWISE_INIT
add_executable       (test_elementwise_init ${PROJECT_SOURCE_DIR}/test_blas/test_elementwise_init.cu)
target_link_libraries(test_elementwise_init qcu cudart cublas)
target_include_directories(test_elementwise_init PUBLIC ${PROJECT_SOURCE_DIR})