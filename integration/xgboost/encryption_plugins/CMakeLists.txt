cmake_minimum_required(VERSION 3.19)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")
cmake_policy(VERSION ${CMAKE_VERSION})
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")


# this has to be set before project()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(xgb_plugins LANGUAGES CUDA CXX VERSION 1.0)
option(BUILD_CUDA_PLUGIN "Build CUDA plugin" ON)
option(BUILD_NVFLARE_PLUGIN "Build NVFlare plugin" ON)

if (BUILD_CUDA_PLUGIN)
  add_subdirectory(cuda_plugin)
else()
  message(STATUS "Skipping CUDA plugin")
endif()

if (BUILD_NVFLARE_PLUGIN)
  add_subdirectory(nvflare_plugin)
else()
  message(STATUS "Skipping NVFLARE plugin")
endif()

