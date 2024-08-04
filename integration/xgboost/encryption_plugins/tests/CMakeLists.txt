file(GLOB_RECURSE TEST_SOURCES "*.cc")

target_sources(xgb_nvflare_test PRIVATE ${TEST_SOURCES})

target_include_directories(xgb_nvflare_test
        PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${xgb_nvflare_SOURCE_DIR/tests}
        ${xgb_nvflare_SOURCE_DIR}/src)

message("Include Dir: ${GTEST_INCLUDE_DIRS}")
target_link_libraries(xgb_nvflare_test
        PRIVATE
        ${GTEST_LIBRARIES})
