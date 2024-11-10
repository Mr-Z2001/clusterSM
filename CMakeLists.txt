cmake_minimum_required(VERSION 3.10)
project(ClusterSM LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.4/bin/nvcc)
set(CMAKE_CUDA_LIBRARY_ARCHITECTURES 86) # A6000
# set(CMAKE_CUDA_LIBRARY_ARCHITECTURES 75) # 2080Ti
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")

include_directories(/usr/local/cuda-12.1/include) # active at school
link_directories(/usr/local/cuda-12.1/lib64) # active at home

# library
add_library(link STATIC

    cpuGraph.h
    cpuGraph.cpp
    gpuGraph.h
    gpuGraph.cpp

    decycle.h
    decycle.cpp

    globals.cuh
    globals.cu

    CLI11.hpp
    cuda_helpers.h

    order.h
    order.cpp

    io.cu
    io.cuh

    structure.cuh
    structure.cu

    filter.cuh
    filter.cu

    order.h
    order.cpp

    join.cuh
    join.cu
)

set_target_properties(link PROPERTIES
    CUDA_ARCHITECTURES "86" 
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(link PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)

# executable
add_executable(ClusterSM main.cu)

set_target_properties(ClusterSM PROPERTIES
    CUDA_ARCHITECTURES "86"
    # CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# link
target_link_libraries(ClusterSM PUBLIC link cudart cuda)

target_compile_options(ClusterSM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)