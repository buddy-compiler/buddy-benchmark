#-------------------------------------------------------------------------------
# Generate MLIRBiquad
#-------------------------------------------------------------------------------

add_custom_command(
  OUTPUT mlir-biquad.o
  COMMAND 
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/BiquadOp/MLIRBiquad.mlir
      -lower-affine 
      -convert-scf-to-cf 
      -llvm-request-c-wrappers
      -convert-arith-to-llvm
      -finalize-memref-to-llvm 
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc 
      -mtriple=${BUDDY_OPT_TRIPLE} 
      -mattr=${BUDDY_OPT_ATTR} 
      -filetype=obj 
      -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/BiquadOp/mlir-biquad.o
  DEPENDS
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate
    ${LLVM_MLIR_BINARY_DIR}/llc 
)

add_library(MLIRBiquad STATIC mlir-biquad.o)
set_target_properties(MLIRBiquad PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Generate dap-op-biquad-benchmark
#-------------------------------------------------------------------------------

add_executable(dap-op-biquad-benchmark Main.cpp)

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
