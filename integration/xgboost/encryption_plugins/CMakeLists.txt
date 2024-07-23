cmake_minimum_required(VERSION 3.19)
project(xgb_nvflare LANGUAGES CXX C VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

option(GOOGLE_TEST "Build google tests" OFF)

file(GLOB_RECURSE LIB_SRC "src/*.cc")

add_library(xgb_nvflare SHARED ${LIB_SRC})
set_target_properties(xgb_nvflare PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  ENABLE_EXPORTS ON
)
target_include_directories(xgb_nvflare PRIVATE ${xgb_nvflare_SOURCE_DIR}/src/include)

if (APPLE)
    add_link_options("LINKER:-object_path_lto,$<TARGET_PROPERTY:NAME>_lto.o")
    add_link_options("LINKER:-cache_path_lto,${CMAKE_BINARY_DIR}/LTOCache")
endif ()

#-- Unit Tests
if(GOOGLE_TEST)
    find_package(GTest REQUIRED)
    enable_testing()
    add_executable(xgb_nvflare_test)
    target_link_libraries(xgb_nvflare_test PRIVATE xgb_nvflare)


    target_include_directories(xgb_nvflare_test PRIVATE ${xgb_nvflare_SOURCE_DIR}/src/include)

    add_subdirectory(${xgb_nvflare_SOURCE_DIR}/tests)

    add_test(
        NAME TestNvflarePlugins
        COMMAND xgb_nvflare_test
        WORKING_DIRECTORY ${xgb_nvflare_BINARY_DIR})

endif()
