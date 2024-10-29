#-------------------------------------------------------------------------------
# MLIR FIR Operation
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-fir.o
        COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/FIROp/MLIRFIR.mlir
        -convert-linalg-to-affine-loops
        -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm -convert-func-to-llvm
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} -filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/FIROp/mlir-fir.o
        )
add_library(MLIRFIR STATIC mlir-fir.o)
set_target_properties(MLIRFIR PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dap-op-fir-benchmark
  BuddyFIRBenchmark.cpp
  Main.cpp
)

target_link_directories(dap-op-fir-benchmark PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(dap-op-fir-benchmark PRIVATE
  # Third-party library
  kfr_io
  # MLIR hand-written benchmark
  MLIRFIR
  # Benchmark library
  GoogleBenchmark
)
