cmake_minimum_required(VERSION 3.18)
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


project(cuda_paillier LANGUAGES CUDA CXX VERSION 1.0)
find_package(gmp REQUIRED)

# Set NVCC compiler
find_program(NVCC nvcc)
if(NOT NVCC)
    message(FATAL_ERROR "NVCC not found! Please make sure CUDA is installed.")
endif()

file(GLOB_RECURSE LIB_SRC
	"src/dam/*.cc"
	"src/plugins/*.cc"
)
message(STATUS "LIB_SRC files: ${LIB_SRC}")

file(GLOB_RECURSE CUDA_SRC
	"CGBN/include/cgbn/cgbn.h"
	"src/cuda-plugin/*.h"
)
message(STATUS "CUDA_SRC files: ${CUDA_SRC}")

set(TARGET_NAME cuda_paillier)

set_source_files_properties(src/plugins/delegated_plugin.cc src/plugins/plugin_main.cc ${CUDA_SRC} PROPERTIES LANGUAGE CUDA)

add_library(${TARGET_NAME} SHARED ${LIB_SRC})
set_target_properties(${TARGET_NAME}
	PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
)
set_target_properties(${TARGET_NAME}
        PROPERTIES
	LINKER_LANGUAGE CUDA
)

target_include_directories(${TARGET_NAME} PRIVATE
	${CMAKE_SOURCE_DIR}/src/include
	${CMAKE_SOURCE_DIR}/src/cuda-plugin
	${CMAKE_SOURCE_DIR}/CGBN/include/cgbn
)

target_compile_features(${TARGET_NAME} PRIVATE cuda_std_17)
target_link_libraries(${TARGET_NAME} PRIVATE gmp::gmpc)
target_link_libraries(${TARGET_NAME} PRIVATE gmp::gmpxx)
target_link_libraries(${TARGET_NAME} PRIVATE gmp::gmp)

