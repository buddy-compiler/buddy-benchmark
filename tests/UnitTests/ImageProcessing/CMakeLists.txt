find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DIPCorr2DTest DIPCorr2DTest.cpp)
target_link_libraries(DIPCorr2DTest ${OpenCV_LIBS} Corr2D Container PNGImage)
