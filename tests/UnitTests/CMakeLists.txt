add_executable(
  test-container
  TestContainer.cpp
)
target_link_libraries(
  test-container
  gtest_main
  ${OpenCV_LIBS}
)

add_dependencies(test-container Container)

include(GoogleTest)
gtest_discover_tests(test-container)
