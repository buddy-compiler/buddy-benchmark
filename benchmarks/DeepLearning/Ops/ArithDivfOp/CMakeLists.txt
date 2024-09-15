# Add custom commands for scalar implementation of arith.divf
add_custom_command(OUTPUT divf_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/ArithDivfOp/ArithDivf.mlir |
          sed 's/@divf/@divf_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O0 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/ArithDivfOp/divf_scalar.o
)
add_library(ArithDivfScalar STATIC divf_scalar.o)
set_target_properties(ArithDivfScalar PROPERTIES LINKER_LANGUAGE CXX)

# Add custom commands for auto-vectorization implementation of arith.divf
add_custom_command(OUTPUT divf_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/ArithDivfOp/ArithDivf.mlir |
          sed 's/@divf/@divf_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/ArithDivfOp/divf_auto_vectorization.o
)
add_library(ArithDivfAutoVectorization STATIC divf_auto_vectorization.o)
set_target_properties(ArithDivfAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

# Add the executable for the benchmark
add_executable(dl-op-linalg-arithdivf-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-arithdivf-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

# Define the benchmark tool
set(BenchmarkTool GoogleBenchmark)

# Link the benchmark executable with the necessary libraries
target_link_libraries(dl-op-linalg-arithdivf-benchmark
  ${BenchmarkTool}
  ArithDivfScalar
  ArithDivfAutoVectorization
)
