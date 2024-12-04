#-------------------------------------------------------------------------------
# Generate MLIRIIRScalar
#-------------------------------------------------------------------------------

add_custom_command(
  OUTPUT iir-scalar.o
  COMMAND 
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/IIROp/MLIRIIRScalar.mlir
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
      -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/IIROp/iir-scalar.o
  DEPENDS
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate
    ${LLVM_MLIR_BINARY_DIR}/llc 
)

add_library(MLIRIIRScalar STATIC iir-scalar.o)
set_target_properties(MLIRIIRScalar PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Generate MLIRIIRVectorization
#-------------------------------------------------------------------------------

add_custom_command(
  OUTPUT iir-vectorization.o
  COMMAND 
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/IIROp/MLIRIIRVectorization.mlir
      -lower-affine
      -convert-scf-to-cf 
      -convert-vector-to-llvm 
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
      -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/IIROp/iir-vectorization.o
  DEPENDS
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate
    ${LLVM_MLIR_BINARY_DIR}/llc 
)

add_library(MLIRIIRVectorization STATIC iir-vectorization.o)
set_target_properties(MLIRIIRVectorization PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Generate dap-op-iir-benchmark
#-------------------------------------------------------------------------------

add_executable(dap-op-iir-benchmark Main.cpp)

target_link_directories(dap-op-iir-benchmark PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(dap-op-iir-benchmark PRIVATE
  # Third-party library
  kfr_io
  # MLIR hand-written benchmark
  MLIRIIRScalar
  MLIRIIRVectorization
  # Buddy DAP library
  BuddyLibDAP
  # LLVM/MLIR library
  StaticMLIRCRunnerUtils
  # Benchmark library
  GoogleBenchmark
)
