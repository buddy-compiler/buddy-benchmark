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

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# BUDDY configuration
#-------------------------------------------------------------------------------
set(BuddyMLIR_DIR ${BUDDY_MLIR_BUILD_DIR}/cmake)
find_package(BuddyMLIR REQUIRED CONFIG)

# BUDDY project.
set(BUDDY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BUDDY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(BUDDY_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/bin)
set(BUDDY_BENCHMARK_DEEP_LEARNING_DIR ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning)

set(BUILD_TESTS OFF CACHE BOOL "Build tests")
set(BUILD_VALIDATION OFF CACHE BOOL "Build validations")
set(BUILD_CORRECTNESS OFF CACHE BOOL "Build correctness checking")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUDDY_BINARY_DIR})

# MLIR binary directory.
set(LLVM_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin)
set(LLVM_MLIR_LIBRARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/lib)

if(CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_ATTR +v,+m CACHE STRING "Target Architecture.")
  set(BUDDY_OPT_TRIPLE riscv64 CACHE STRING "Target Triple.")
  set(BUDDY_LIB_DIR ${BUDDY_MLIR_BUILD_CROSS_DIR}/lib)
  set(OMP_SHARED_DIR ${RISCV_OMP_SHARED})
else()
  set(BUDDY_OPT_ATTR avx512f CACHE STRING "Target Architecture.")
  set(BUDDY_OPT_TRIPLE x86_64-unknown-linux-gnu CACHE STRING "Target Triple.")
  set(BUDDY_LIB_DIR ${BUDDY_MLIR_BUILD_DIR}/lib)
  set(OMP_SHARED_DIR ${LLVM_MLIR_LIBRARY_DIR}/libomp.so)
endif()
message(STATUS "Configuring Target Architecture: ${BUDDY_OPT_ATTR}")
message(STATUS "Configuring Target Triple: ${BUDDY_OPT_TRIPLE}")

# Add BUDDY files to the include path
include_directories(${BUDDY_MAIN_INCLUDE_DIR})
include_directories(${BUDDY_MLIR_INTERFACE_DIR})
include_directories(${BUDDY_THIRDPARTY_INCLUDE_DIR})

# Helper functions.
include(${BUDDY_SOURCE_DIR}/cmake/buddy-benchmark.cmake)

# -------------------------------------------------------------------------------
# Deploy google/benchmark
# -------------------------------------------------------------------------------
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
  -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/vendor/benchmark
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DBENCHMARK_ENABLE_TESTING=OFF
  -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
  -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
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

# -------------------------------------------------------------------------------
# Find OpenCV
# -------------------------------------------------------------------------------
if(DEFINED IMAGE_PROCESSING_BENCHMARKS)
  find_package(OpenCV REQUIRED CONFIG)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# -------------------------------------------------------------------------------
# Find PNG
# -------------------------------------------------------------------------------
if(DEFINED IMAGE_PROCESSING_BENCHMARKS)
  find_package(PNG REQUIRED)
  include_directories(${PNG_INCLUDE_DIR})
endif()

# -------------------------------------------------------------------------------
# Hardware detection
# -------------------------------------------------------------------------------
include(${BUDDY_SOURCE_DIR}/cmake/check-simd.cmake)
check_simd()

# -------------------------------------------------------------------------------
# Subdirectory
# -------------------------------------------------------------------------------
add_subdirectory(benchmarks)
add_subdirectory(utils)

if(BUILD_VALIDATION)
  add_subdirectory(validation)
endif()
