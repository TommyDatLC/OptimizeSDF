# Khuyến nghị để bản 3.24 trở lên vì bản 4.1 chưa chính thức ra mắt rộng rãi
cmake_minimum_required(VERSION 3.24)

project(OptimizeSDF LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG master # Nên thêm tag (ví dụ: master hoặc v1.3.0) để tránh cảnh báo của CMake
)
FetchContent_MakeAvailable(polyscope)

# ==============================================================================
# BƯỚC 1: KHAI BÁO TARGET (BẮT BUỘC PHẢI ĐẶT TRƯỚC)
# ==============================================================================
add_executable(OptimizeSDF
        main.cu
        Core/Matrix.cu
        Core/Matrix.cuh
        Core/MatrixMemoryManager.cu
        Core/MatrixMemoryManager.cuh
        Core/MatrixUtils.cuh
        Core/ClipSpaceConversion.hpp
        Core/mathHelper.cu
        Core/Model.cu
        Core/Model.cuh
        Core/ModelHelper.cu
        NormalSDFOptix/SDFMain.cpp

)
add_library(OptixShaders OBJECT NormalSDFOptix/SDFOptix.cu)

# 2. Bật cờ PTX Compilation: Báo cho NVCC biết chỉ dịch ra PTX rồi dừng lại
set_property(TARGET OptixShaders PROPERTY CUDA_PTX_COMPILATION ON)

# 3. Đảm bảo Shader này đọc được các file header của OptiX và thư mục Core
target_include_directories(OptixShaders PRIVATE
        /home/datdau/CUDA/OptimizeSDF/OptiX
        ${CMAKE_CURRENT_SOURCE_DIR}/Core
)

# 4. Bắt buộc project chính phải đợi file PTX compile xong mới được buildc
add_dependencies(OptimizeSDF OptixShaders)
# 1. Chỉ định thư mục Include cho OptiX
target_include_directories(OptimizeSDF PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/OptiX)

# 2. BẬT Tính năng CUDA Separable Compilation và Architecture
# - CUDA_SEPARABLE_COMPILATION: Bắt buộc để link các hàm device giữa các file .cu
# - CUDA_ARCHITECTURES: Rất quan trọng khi dùng OptiX/CUDA đời mới.
#   (Ví dụ: 86 là kiến trúc Ampere cho RTX 30-series. Hãy đổi thành 89 nếu dùng RTX 40-series, hoặc 75 nếu dùng RTX 20-series)
set_target_properties(OptimizeSDF PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# 3. Link các thư viện cần thiết
target_link_libraries(OptimizeSDF
        PRIVATE
        cublas
        polyscope
)

# ==============================================================================
# BƯỚC 3: CÁC THAO TÁC KHÁC
# ==============================================================================
file(COPY "Model" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})