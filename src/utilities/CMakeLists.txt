# Set the project name for the utilities library
project(Utilities LANGUAGES CXX CUDA)

# Collect all source files in the utilities directory
file(GLOB UTILITIES_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Create a static or shared library (choose either STATIC or SHARED)
add_library(utilities STATIC ${UTILITIES_SOURCES})

# Optionally, set CUDA properties if needed
set_target_properties(utilities PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include directories for the utilities (so other projects can use its headers)
target_include_directories(utilities PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
