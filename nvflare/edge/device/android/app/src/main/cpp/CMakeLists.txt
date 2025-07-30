cmake_minimum_required(VERSION 3.18.1)

project("executorch_training")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_library(log-lib log)

# Add ExecuTorch include directories
# These paths should be updated based on your ExecuTorch installation
set(EXECUTORCH_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../executorch/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../executorch/extension/include
)

# Add ExecuTorch library directories
set(EXECUTORCH_LIB_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../executorch/build/lib
)

# Include directories
include_directories(${EXECUTORCH_INCLUDE_DIRS})

# Link directories
link_directories(${EXECUTORCH_LIB_DIRS})

# Create the shared library
add_library(
    executorch_training
    SHARED
    executorch_training.cpp
)

# Link libraries
target_link_libraries(
    executorch_training
    ${log-lib}
    # Add ExecuTorch libraries here
    # executorch_core
    # executorch_extension
    # executorch_training
)

# Set compile definitions
target_compile_definitions(executorch_training PRIVATE
    ANDROID_PLATFORM=1
)

# Set compile options
target_compile_options(executorch_training PRIVATE
    -Wall
    -Wextra
    -fexceptions
    -frtti
) 