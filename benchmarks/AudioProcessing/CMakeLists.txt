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
set(LLVM_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin)

message(STATUS "Configuring Stride Size: ${SPLITING_SIZE}")
message(STATUS "Configuring MLIR Linalg Tile Size: ${MLIR_LINALG_TILE}")

#-------------------------------------------------------------------------------
# Audio Processing Benchmark Target
#-------------------------------------------------------------------------------

add_subdirectory(${KFR_DIR} ./kfr)
include_directories(${KFR_DIR}/include)

#-------------------------------------------------------------------------------
# MLIR Biquad Operation
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-biquad.o
        COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/MLIRBiquad.mlir
        -convert-linalg-to-affine-loops
        -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
        --llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/mlir-biquad.o
        )
add_library(MLIRBiquad STATIC mlir-biquad.o)
set_target_properties(MLIRBiquad PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR IIR Operation
#-------------------------------------------------------------------------------

# TODO:
# Support only when input audio size is multiple of splitting size.
# Add a support for all input audio size.
add_custom_command(OUTPUT mlir-iir.o
        COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/MLIRIir.mlir
        -lower-dap="DAP-vector-splitting=64" -convert-linalg-to-affine-loops
        -lower-affine
        -convert-scf-to-cf 
        -convert-vector-to-llvm 
        --llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/mlir-iir.o
        )
add_library(MLIRIir STATIC mlir-iir.o)
set_target_properties(MLIRIir PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT iir-vectorization.o
        COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/MLIRIirVectorization.mlir
        -convert-linalg-to-affine-loops
        -lower-affine
        -convert-scf-to-cf 
        -convert-vector-to-llvm 
        --llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/iir-vectorization.o
        )
add_library(MLIRIirVectorization STATIC iir-vectorization.o)
set_target_properties(MLIRIirVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(audio-processing-benchmark
  KFRFft.cpp
  KFRBiquad.cpp
  BuddyBiquadBenchmark.cpp
  KFRIir.cpp
  BuddyIirBenchmark.cpp
  Main.cpp
)

target_link_directories(audio-processing-benchmark
  PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(audio-processing-benchmark
  PRIVATE
  kfr_io
  kfr_dft
  MLIRBiquad
  MLIRIir
  MLIRIirVectorization
  BuddyLibDAP
  BuddyLibDAPVectorization
  GoogleBenchmark
)
