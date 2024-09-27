# Specify the project name for this subproject
project(Swizzle LANGUAGES CXX CUDA)

# Add an executable for this subproject
add_executable(swizzle swizzle.cu)

# Link necessary libraries (e.g., CUDA)
target_link_libraries(swizzle PRIVATE cudart)

# Optionally, set CUDA properties like separable compilation
set_target_properties(swizzle PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optionally, set include directories for this subproject
target_include_directories(swizzle PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
