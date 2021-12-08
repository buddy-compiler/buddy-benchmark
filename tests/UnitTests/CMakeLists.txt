#enable_testing()

add_executable(
  TestContainer
  TestContainer.cpp
)
target_link_libraries(
  TestContainer
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(TestContainer)
