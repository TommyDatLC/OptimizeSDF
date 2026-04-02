cmake_minimum_required(VERSION 4.1)
project(OptimizeSDF LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD  20)

include(FetchContent)
FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG
)
FetchContent_MakeAvailable(polyscope)
add_executable(OptimizeSDF main.cu
        Matrix.cu
        Matrix.cuh
        MatrixMemoryManager.cu
        MatrixMemoryManager.cuh
        MatrixUtils.cuh
        ClipSpaceConversion.hpp
        mathHelper.cu
        Model.cpp
        Model.h
        ModelHelper.cu
)
target_link_libraries(OptimizeSDF
        PRIVATE
        cublas
        polyscope
)

set_target_properties(OptimizeSDF PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
file(COPY "Model" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})