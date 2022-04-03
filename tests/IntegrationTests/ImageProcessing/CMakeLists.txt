find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DIPCorr2DTest DIPCorr2DTest.cpp)

include(GoogleTest)
gtest_discover_tests(DIPCorr2DTest)

target_link_libraries(DIPCorr2DTest ${OpenCV_LIBS} BuddyCorr2D Container PNGImage gtest_main)
