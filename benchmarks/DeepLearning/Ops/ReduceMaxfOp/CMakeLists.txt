separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# Add custom commands for scalar implementation of reduce.maxf
add_custom_command(OUTPUT maxf_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/ReduceMaxfOp/ReduceMaxf.mlir |
          sed 's/@maxf/@maxf_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o maxf_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} maxf_scalar.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/maxf_scalar.o
)
add_library(ReduceMaxfScalar STATIC maxf_scalar.o)
set_target_properties(ReduceMaxfScalar PROPERTIES LINKER_LANGUAGE CXX)

# Add custom commands for auto-vectorization implementation of reduce.maxf
add_custom_command(OUTPUT maxf_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/ReduceMaxfOp/ReduceMaxf.mlir |
          sed 's/@maxf/@maxf_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o maxf_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} maxf_auto_vectorization.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/maxf_auto_vectorization.o
)
add_library(ReduceMaxfAutoVectorization STATIC maxf_auto_vectorization.o)
set_target_properties(ReduceMaxfAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

# Add the executable for the benchmark
add_executable(dl-op-linalg-reducemaxf-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-reducemaxf-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

# Define the benchmark tool
set(BenchmarkTool GoogleBenchmark)

# Link the benchmark executable with the necessary libraries
target_link_libraries(dl-op-linalg-reducemaxf-benchmark
  ${BenchmarkTool}
  ReduceMaxfScalar
  ReduceMaxfAutoVectorization
)
