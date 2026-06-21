cmake_minimum_required(VERSION 3.24)
project(FastRobustSDF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation flags
if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

include(FetchContent)
FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG master
)
FetchContent_MakeAvailable(polyscope)

add_executable(FastRobustSDF
    main.cpp
    sdf_offset.cpp
)

target_include_directories(FastRobustSDF PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(FastRobustSDF PRIVATE polyscope)

# Copy Model/ into build directory
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../Model" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
