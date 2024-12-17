cmake_minimum_required(VERSION 3.8)
project(QCU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
# set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)  
set(CMAKE_CUDA_ARCHITECTURES 70)    # for 4060 laptop
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -G")
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G") # -G 表示生成GPU调试信息，-O0 禁止优化以便于调试
# set(CMAKE_BUILD_TYPE Debug) # 设定默认构建类型为Debug

set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")

add_subdirectory(external/qcu_io)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/qcu_base QCU_BASE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/qcu_blas QCU_BLAS)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/qcd QCU_QCD)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/communication QCU_COMMUNICATION)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/solver QCU_SOLVER)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/qcu_config QCU_CONFIG)

# shared library
add_library(qcu 
    SHARED
    ${PROJECT_SOURCE_DIR}/src/interface.cu
    ${PROJECT_SOURCE_DIR}/src/qcu_interface.cu
	${QCU_QCD}
    ${PROJECT_SOURCE_DIR}/src/data_format/qcu_data_format_shift.cu
	${QCU_BASE}
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

target_link_directories(qcu PUBLIC ${PROJECT_BINARY_DIR}/external/qcu_io)
target_link_libraries(qcu
	qcu_io
	cudart
	cudadevrt
    cublas
	${MPI_LIBRARIES}
)


add_subdirectory(tests)
