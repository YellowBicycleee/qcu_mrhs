cmake_minimum_required(VERSION 3.16)

project(QCU LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
#find_library(NVTX_LIBRARY nvToolsExt REQUIRED)
#target_link_libraries(your_target PRIVATE ${NVTX_LIBRARY})

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v --keep") #保留ptx
# include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)  
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CUDA_ARCHITECTURES 89)    # for 4060 laptop
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) #生成compile_commands.json


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
    ${MPI_INCLUDE_PATH}
    /usr/local/cuda/bin/../targets/x86_64-linux/include
)

target_link_directories(qcu PUBLIC ${PROJECT_BINARY_DIR}/external/qcu_io)
target_link_libraries(qcu
	qcu_io
	cudart
	cudadevrt
    cublas
	nvToolsExt
	${MPI_LIBRARIES}
)
#target_link_libraries(qcu PRIVATE -lnvToolsExt)


message("MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
add_subdirectory(tests)
