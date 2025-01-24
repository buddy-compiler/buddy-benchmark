separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# Add custom commands for scalar implementation of arith.negf
add_custom_command(OUTPUT negf_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/ArithNegfOp/ArithNegf.mlir |
          sed 's/@negf/@negf_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o negf_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} negf_scalar.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/negf_scalar.o
)
add_library(ArithNegfScalar STATIC negf_scalar.o)
set_target_properties(ArithNegfScalar PROPERTIES LINKER_LANGUAGE CXX)

# Add custom commands for auto-vectorization implementation of arith.negf
add_custom_command(OUTPUT negf_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/ArithNegfOp/ArithNegf.mlir |
          sed 's/@negf/@negf_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o negf_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} negf_auto_vectorization.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/negf_auto_vectorization.o
)
add_library(ArithNegfAutoVectorization STATIC negf_auto_vectorization.o)
set_target_properties(ArithNegfAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

# Add the executable for the benchmark
add_executable(dl-op-linalg-arithnegf-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-arithnegf-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

# Define the benchmark tool
set(BenchmarkTool GoogleBenchmark)

# Link the benchmark executable with the necessary libraries
target_link_libraries(dl-op-linalg-arithnegf-benchmark
  ${BenchmarkTool}
  ArithNegfScalar
  ArithNegfAutoVectorization
)
