
option(GOOGLE_TEST "Build google tests" OFF)

file(GLOB_RECURSE LIB_SRC
	${CMAKE_SOURCE_DIR}/shared/dam/*.cc
	${CMAKE_SOURCE_DIR}/shared/plugins/*.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
)
message(STATUS "LIB_SRC files: ${LIB_SRC}")

set(TARGET_NAME nvflare)
add_library(${TARGET_NAME} SHARED ${LIB_SRC})
set_target_properties(${TARGET_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  ENABLE_EXPORTS ON
)
target_include_directories(${TARGET_NAME} PRIVATE
	${CMAKE_SOURCE_DIR}/shared/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (APPLE)
    add_link_options("LINKER:-object_path_lto,$<TARGET_PROPERTY:NAME>_lto.o")
    add_link_options("LINKER:-cache_path_lto,${CMAKE_BINARY_DIR}/LTOCache")
endif ()

#-- Unit Tests
if(GOOGLE_TEST)
    find_package(GTest REQUIRED)
    enable_testing()
    add_executable(nvflare_test)
    target_link_libraries(nvflare_test PRIVATE ${TARGET_NAME})


    target_include_directories(nvflare_test PRIVATE
        ${CMAKE_SOURCE_DIR}/shared/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)

    add_test(
        NAME TestNvflarePlugins
        COMMAND nvflare_test
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

endif()
