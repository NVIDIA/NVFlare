file(GLOB_RECURSE TEST_SOURCES "*.cc")

target_sources(proc_test PRIVATE ${TEST_SOURCES})

target_include_directories(proc_test
        PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${proc_nvflare_SOURCE_DIR/tests}
        ${proc_nvflare_SOURCE_DIR}/src)

message("Include Dir: ${GTEST_INCLUDE_DIRS}")
target_link_libraries(proc_test
        PRIVATE
        ${GTEST_LIBRARIES})
