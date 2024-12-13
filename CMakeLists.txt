cmake_minimum_required(VERSION 3.18)

# Set the project name and version
project(MCT VERSION 1.0 LANGUAGES CXX CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 61 75 86)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optionally, include CUDA directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add the utilities directory and build the utilities library
add_subdirectory(src/utilities)

# Add subdirectories (each of these subdirectories contains its own CMakeLists.txt)
add_subdirectory(src/swizzle)

# Optionally, if you want to create a shared library or utility to be used by all subprojects
# add_library(MySharedLib src/shared_code.cu src/shared_code.h)
# target_link_libraries(MySharedLib ...)
