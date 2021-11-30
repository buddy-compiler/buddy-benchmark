enable_testing()

add_executable(
  fake_test
  container.cpp
)
target_link_libraries(
  fake_test
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(fake_test)
