# Specify the project name for this subproject
project(Swizzling LANGUAGES CXX CUDA)

# Collect source files for this subproject
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCES
    *.cpp
    *.cu
    *.h
    *.cuh
)

# Add an executable for this subproject
add_executable(swizzling ${PROJECT_NAME}_SOURCES)

# Link necessary libraries (e.g., CUDA)
target_link_libraries(swizzling PRIVATE CUDA::cudart)

# Optionally, set CUDA properties like separable compilation
set_target_properties(swizzling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optionally, set include directories for this subproject
target_include_directories(swizzling PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
