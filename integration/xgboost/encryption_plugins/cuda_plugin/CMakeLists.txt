set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(gmp REQUIRED)

# Set NVCC compiler
find_program(NVCC nvcc)
if(NOT NVCC)
    message(FATAL_ERROR "NVCC not found! Please make sure CUDA is installed.")
endif()

file(GLOB_RECURSE LIB_SRC
	${CMAKE_SOURCE_DIR}/shared/dam/*.cc
	${CMAKE_SOURCE_DIR}/shared/plugins/*.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
)
message(STATUS "LIB_SRC files: ${LIB_SRC}")

file(GLOB_RECURSE CUDA_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/CGBN/include/cgbn/cgbn.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)
message(STATUS "CUDA_SRC files: ${CUDA_SRC}")

set(TARGET_NAME cuda_paillier)

set_source_files_properties(
	${CMAKE_CURRENT_SOURCE_DIR}/src/delegated_plugin.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/plugin_main.cc
	${CUDA_SRC}
	PROPERTIES LANGUAGE CUDA
)

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
	${CMAKE_SOURCE_DIR}/shared/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/CGBN/include/cgbn
)

target_compile_features(${TARGET_NAME} PRIVATE cuda_std_17)
target_link_libraries(${TARGET_NAME} PRIVATE gmp::gmpc)
target_link_libraries(${TARGET_NAME} PRIVATE gmp::gmpxx)
target_link_libraries(${TARGET_NAME} PRIVATE gmp::gmp)

