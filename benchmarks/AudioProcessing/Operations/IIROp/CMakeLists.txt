#-------------------------------------------------------------------------------
# MLIR IIR Operation
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-iir.o
        COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/IIROp/MLIRIIR.mlir
        -lower-dap="DAP-vector-splitting=64" -convert-linalg-to-affine-loops
        -lower-affine
        -convert-scf-to-cf 
        -convert-vector-to-llvm 
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} -filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/IIROp/mlir-iir.o
        )
add_library(MLIRIIR STATIC mlir-iir.o)
set_target_properties(MLIRIIR PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT iir-vectorization.o
        COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/IIROp/MLIRIIRVectorization.mlir
        -convert-linalg-to-affine-loops
        -lower-affine
        -convert-scf-to-cf 
        -convert-vector-to-llvm 
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} -filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/IIROp/iir-vectorization.o
        )
add_library(MLIRIIRVectorization STATIC iir-vectorization.o)
set_target_properties(MLIRIIRVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dap-op-iir-benchmark
  KFRIIR.cpp
  BuddyIIRBenchmark.cpp
  Main.cpp
)

target_link_directories(dap-op-iir-benchmark PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(dap-op-iir-benchmark PRIVATE
  # Third-party library
  kfr_io
  # MLIR hand-written benchmark
  MLIRIIR
  MLIRIIRVectorization
  # Buddy DAP library
  BuddyLibDAP
  # LLVM/MLIR library
  StaticMLIRCRunnerUtils
  # Benchmark library
  GoogleBenchmark
)
