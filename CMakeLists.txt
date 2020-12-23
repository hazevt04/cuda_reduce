# CMakeLists.txt to build add.cu
cmake_minimum_required(VERSION 2.8)

project(cuda_reduce)

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA QUIET REQUIRED)

# Selects GPU arch flags for nvcc based on target_CUDA_architectures
# Xavier is compute capability 7.2, Tegra X2 is 6.1
cuda_select_nvcc_arch_flags(ARCH_FLAGS 7.2 6.1)

LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")

set(CUDA_VERBOSE_BUILD ON)

set(SRCS
   "src/my_cufft_utils.cpp"
   "src/my_utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/cuda_reduce.cu"
)

include_directories( "include" )

cuda_add_executable(${EXEC_NAME}
   "${SRCS}"
)

