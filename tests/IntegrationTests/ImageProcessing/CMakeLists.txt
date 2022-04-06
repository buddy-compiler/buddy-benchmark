find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(
  image-processing-integration-tests
  DIPCorr2DTest.cpp
)

include(GoogleTest)
gtest_discover_tests(image-processing-integration-tests)

target_link_libraries(
  image-processing-integration-tests 
  ${OpenCV_LIBS} 
  BuddyCorr2D 
  Container 
   
  gtest_main
)
