find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "Configuring benchmarks: google")

#-----------------------------------------------------------------------------
# Deploy google/benchmark
#-----------------------------------------------------------------------------
include(ExternalProject)

ExternalProject_Add(project_googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG "v1.6.0"
  GIT_SHALLOW 1
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
  TIMEOUT 10
  BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBENCHMARK_ENABLE_TESTING=OFF
  UPDATE_COMMAND ""
  TEST_COMMAND "")

ExternalProject_Get_Property(project_googlebenchmark INSTALL_DIR)

file(MAKE_DIRECTORY ${INSTALL_DIR}/include)
add_library(buddy_googlebenchmark STATIC IMPORTED)
target_include_directories(buddy_googlebenchmark INTERFACE ${INSTALL_DIR}/include)
set_property(TARGET buddy_googlebenchmark PROPERTY IMPORTED_LOCATION
  "${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}")

add_dependencies(buddy_googlebenchmark project_googlebenchmark)

find_package(Threads)
target_link_libraries(buddy_googlebenchmark INTERFACE Threads::Threads)

add_executable(OpenCVBenchmark OpenCVBenchmark.cpp)
target_link_libraries(OpenCVBenchmark ${OpenCV_LIBS} buddy_googlebenchmark)

add_executable(BuddyBenchmark BuddyBenchmark.cpp)
add_dependencies(BuddyBenchmark conv-opt)
target_link_libraries(BuddyBenchmark ${OpenCV_LIBS} Conv2D buddy_googlebenchmark)

add_executable(CompareBuddyOpencv CompareBuddyOpencv.cpp)
add_dependencies(CompareBuddyOpencv conv-opt)
target_link_libraries(CompareBuddyOpencv ${OpenCV_LIBS} Conv2D buddy_googlebenchmark)
