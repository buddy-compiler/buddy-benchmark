##===- CMakeLists.txt - buddy-benchmark cmake root ------------*- cmake -*-===//
##
## Configure the buddy-benchmark build.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(buddy-benchmark LANGUAGES CXX C)
  
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_FLAGS "-no-pie")
set(CMAKE_C_FLAGS "-no-pie")

#-------------------------------------------------------------------------------
# BUDDY configuration
#-------------------------------------------------------------------------------
set(BuddyMLIR_DIR ${BUDDY_MLIR_BUILD_DIR}/cmake)
find_package(BuddyMLIR REQUIRED CONFIG)

# BUDDY project.
set(BUDDY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUDDY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(BUDDY_EXAMPLES_DIR ${BUDDY_SOURCE_DIR}/examples)
set(BUDDY_OPT_ATTR avx512f CACHE STRING "Target Architecture.")
set(BUDDY_OPT_TRIPLE x86_64-unknown-linux-gnu CACHE STRING "Target Triple.")
message(STATUS "Configuring Target Architecture: ${BUDDY_OPT_ATTR}")
message(STATUS "Configuring Target Triple: ${BUDDY_OPT_TRIPLE}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUDDY_BINARY_DIR})

# Add BUDDY files to the include path
include_directories(${BUDDY_MAIN_INCLUDE_DIR})
include_directories(${BUDDY_MLIR_INTERFACE_DIR})
include_directories(${BUDDY_MLIR_LIB_DIR})

# MLIR binary directory.
set(LLVM_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin)

if(CMAKE_TOOLCHAIN_FILE STREQUAL ${BUDDY_SOURCE_DIR}/cmake/riscv-toolchain.cmake)
  # Use cross-compiled RISC-V MLIR library.
  set(LLVM_MLIR_LIBRARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build-cross-mlir/lib)
else()
  set(LLVM_MLIR_LIBRARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/lib)
endif()

# Helper functions.
include(${BUDDY_SOURCE_DIR}/cmake/buddy-benchmark.cmake)

#-------------------------------------------------------------------------------
# Deploy google/benchmark
#-------------------------------------------------------------------------------

message(STATUS "Configuring benchmarks: google")

include(ExternalProject)

ExternalProject_Add(project_googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG "v1.6.0"
  GIT_SHALLOW 1
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
  TIMEOUT 10
  BUILD_BYPRODUCTS <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}
  CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DBENCHMARK_ENABLE_TESTING=OFF
  UPDATE_COMMAND ""
  TEST_COMMAND "")

ExternalProject_Get_Property(project_googlebenchmark INSTALL_DIR)

file(MAKE_DIRECTORY ${INSTALL_DIR}/include)
add_library(GoogleBenchmark STATIC IMPORTED)
target_include_directories(GoogleBenchmark INTERFACE ${INSTALL_DIR}/include)
set_property(TARGET GoogleBenchmark PROPERTY IMPORTED_LOCATION
  "${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX}")

add_dependencies(GoogleBenchmark project_googlebenchmark)

find_package(Threads)
target_link_libraries(GoogleBenchmark INTERFACE Threads::Threads)

#-------------------------------------------------------------------------------
# Find OpenCV
#-------------------------------------------------------------------------------

if(DEFINED IMAGE_PROCESSING_BENCHMARKS OR DEEP_LEARNING_BENCHMARKS OR OP_OPTIMIZATION_BENCHMARKS)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

#-------------------------------------------------------------------------------
# Find PNG
#-------------------------------------------------------------------------------

if(DEFINED IMAGE_PROCESSING_BENCHMARKS OR DEEP_LEARNING_BENCHMARKS)
  find_package(PNG REQUIRED)
  include_directories(${PNG_INCLUDE_DIR})
endif()

#-------------------------------------------------------------------------------
# Hardware detection
#-------------------------------------------------------------------------------

include(${BUDDY_SOURCE_DIR}/cmake/check-simd.cmake)
check_simd()

#-------------------------------------------------------------------------------
# Subdirectory
#-------------------------------------------------------------------------------

add_subdirectory(benchmarks)
add_subdirectory(utils)
