add_executable(
  test-container
  TestContainer.cpp
)
target_link_libraries(
  test-container
  gtest_main
)

add_dependencies(test-container Container)

include(GoogleTest)
gtest_discover_tests(test-container)

add_subdirectory(ImageProcessing)
