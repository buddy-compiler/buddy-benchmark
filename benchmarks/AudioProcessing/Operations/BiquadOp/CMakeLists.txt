#-------------------------------------------------------------------------------
# MLIR Biquad Operation
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-biquad.o
        COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/BiquadOp/MLIRBiquad.mlir
        -convert-linalg-to-affine-loops
        -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} -filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/BiquadOp/mlir-biquad.o
        )
add_library(MLIRBiquad STATIC mlir-biquad.o)
set_target_properties(MLIRBiquad PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dap-op-biquad-benchmark
  KFRBiquad.cpp
  BuddyBiquadBenchmark.cpp
  Main.cpp
)

target_link_directories(dap-op-biquad-benchmark PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(dap-op-biquad-benchmark PRIVATE
  # Third-party library
  kfr_io
  # MLIR hand-written benchmark
  MLIRBiquad
  # Buddy DAP library
  BuddyLibDAP
  # LLVM/MLIR library
  StaticMLIRCRunnerUtils
  # Benchmark library
  GoogleBenchmark
)
