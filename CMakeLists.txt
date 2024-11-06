cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
set(CMAKE_CUDA_ARCHITECTURES 89)    # for 4060 laptop
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -G")
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0") # -G 表示生成GPU调试信息，-O0 禁止优化以便于调试
# set(CMAKE_BUILD_TYPE Debug) # 设定默认构建类型为Debug

project(QCU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}
find_package(MPI REQUIRED)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/qcu_blas QCU_BLAS)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/communication QCU_COMMUNICATION)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/solver QCU_SOLVER)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/qcu_config QCU_CONFIG)

# shared library
add_library(qcu 
    SHARED
    ${PROJECT_SOURCE_DIR}/src/interface.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_interface.cu  
    ${PROJECT_SOURCE_DIR}/src/qcd/qcu_wilson_dslash.cu
    ${PROJECT_SOURCE_DIR}/src/data_format/qcu_data_format_shift.cu
    ${QCU_BLAS}
    ${QCU_SOLVER}
	${QCU_CONFIG}
	${QCU_COMMUNICATION}
)
target_include_directories(qcu 
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/qcu_io/include
    /usr/local/cuda/bin/../targets/x86_64-linux/include
)

target_link_directories(qcu PUBLIC ${PROJECT_SOURCE_DIR}/external/qcu_io/build)
target_link_libraries(qcu
    qcu_io
	cudart
	cudadevrt
    cublas
	${MPI_LIBRARIES}
	# nccl
)


add_subdirectory(tests)
