cmake_minimum_required(VERSION 4.1)
project(OptimizeSDF CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(OptimizeSDF main.cu
        Matrix.cpp
        Matrix.h
        MatrixMemoryManager.cpp
        MatrixMemoryManager.h
        MatrixUtils.cpp
        ClipSpaceConversion.cpp
        mathHelper.cpp
)
target_link_libraries(OptimizeSDF
        cublas
)
set_target_properties(OptimizeSDF PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
file(GLOB_RECURSE SOURCES "360.obj")