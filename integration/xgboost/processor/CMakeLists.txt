cmake_minimum_required(VERSION 3.19)
project(proc_nvflare LANGUAGES CXX C VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

option(GOOGLE_TEST "Build google tests" OFF)

file(GLOB_RECURSE LIB_SRC
        "src/*.h"
        "src/*.cc"
        )

add_library(proc_nvflare SHARED ${LIB_SRC})
set(XGB_SRC ${proc_nvflare_SOURCE_DIR}/../../../../xgboost)
target_include_directories(proc_nvflare PRIVATE ${proc_nvflare_SOURCE_DIR}/src/include
        ${XGB_SRC}/src
        ${XGB_SRC}/rabit/include
        ${XGB_SRC}/include
        ${XGB_SRC}/dmlc-core/include)

link_directories(${XGB_SRC}/lib/)

if (APPLE)
    add_link_options("LINKER:-object_path_lto,$<TARGET_PROPERTY:NAME>_lto.o")
    add_link_options("LINKER:-cache_path_lto,${CMAKE_BINARY_DIR}/LTOCache")
endif ()

target_link_libraries(proc_nvflare ${XGB_SRC}/lib/libxgboost${CMAKE_SHARED_LIBRARY_SUFFIX})

#-- Unit Tests
if(GOOGLE_TEST)
    find_package(GTest REQUIRED)
    enable_testing()
    add_executable(proc_test)
    target_link_libraries(proc_test PRIVATE proc_nvflare)


    target_include_directories(proc_test PRIVATE ${proc_nvflare_SOURCE_DIR}/src/include
        ${XGB_SRC}/src
        ${XGB_SRC}/rabit/include
        ${XGB_SRC}/include
        ${XGB_SRC}/dmlc-core/include
        ${XGB_SRC}/tests)

    add_subdirectory(${proc_nvflare_SOURCE_DIR}/tests)

    add_test(
        NAME TestProcessor
        COMMAND proc_test
        WORKING_DIRECTORY ${proc_nvflare_BINARY_DIR})

endif()