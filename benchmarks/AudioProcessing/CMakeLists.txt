#-------------------------------------------------------------------------------
# Optimization Configurations
#-------------------------------------------------------------------------------

if (${BUDDY_OPT_STRIP_MINING})
  set(SPLITING_SIZE ${BUDDY_OPT_STRIP_MINING} CACHE STRING "Spliting Size")
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256 CACHE STRING "Spliting Size")
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128 CACHE STRING "Spliting Size")
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
elseif(HAVE_NEON)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
else()  # Fallback value, avoid crashing
  set(SPLITING_SIZE 16 CACHE STRING "Spliting Size")
endif()

set(MLIR_LINALG_TILE 2 CACHE STRING "MLIR Linalg Tile Size")
set(LLVM_MLIR_BINARY_DIR ${BUDDY_OPT_BUILD_DIR}/../llvm/build/bin)

message(STATUS "Configuring Stride Size: ${SPLITING_SIZE}")
message(STATUS "Configuring MLIR Linalg Tile Size: ${MLIR_LINALG_TILE}")

#-------------------------------------------------------------------------------
# Audio Processing Benchmark Target
#-------------------------------------------------------------------------------

add_subdirectory(${KFR_DIR} ./kfr)
include_directories(${KFR_DIR}/include)

#-------------------------------------------------------------------------------
# DAP Dialect Buddy Biquad Operation
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT buddy-biquad.o
        COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/BuddyBiquad.mlir
        -lower-dap="DAP-vector-splitting=${SPLITING_SIZE}" -convert-linalg-to-affine-loops
        -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
        --llvm-request-c-wrappers
        -convert-arith-to-llvm
        -convert-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/buddy-biquad.o
        )
add_library(BuddyBiquad STATIC buddy-biquad.o)
set_target_properties(BuddyBiquad PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# DAP Dialect Buddy IIR Operation
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT buddy-iir.o
        COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/BuddyIir.mlir
        -lower-dap="DAP-vector-splitting=${SPLITING_SIZE}" -convert-linalg-to-affine-loops
        -lower-affine
        -convert-scf-to-cf 
        -convert-vector-to-llvm 
        --llvm-request-c-wrappers
        -convert-arith-to-llvm
        -convert-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/buddy-iir.o
        )
add_library(BuddyIir STATIC buddy-iir.o)
set_target_properties(BuddyIir PROPERTIES LINKER_LANGUAGE CXX)

add_executable(audio-processing-benchmark
  KFRFft.cpp
  KFRFir.cpp 
  KFRIir.cpp
  KFRBiquad.cpp
  BuddyBiquadBenchmark.cpp
  BuddyIirBenchmark.cpp
  Main.cpp
)

target_link_directories(audio-processing-benchmark PRIVATE ${KFR_DIR}/build/)
target_link_libraries(audio-processing-benchmark
  PRIVATE
  kfr_io
  kfr_dft
  BuddyBiquad
  BuddyIir
  GoogleBenchmark
)
