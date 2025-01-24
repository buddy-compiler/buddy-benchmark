separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# Add custom commands for scalar implementation of math.rsqrt
add_custom_command(OUTPUT rsqrt_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MathRsqrtOp/MathRsqrt.mlir |
          sed 's/@rsqrt/@rsqrt_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o rsqrt_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} rsqrt_scalar.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/rsqrt_scalar.o
)
add_library(MathRsqrtScalar STATIC rsqrt_scalar.o)
set_target_properties(MathRsqrtScalar PROPERTIES LINKER_LANGUAGE CXX)

# Add custom commands for auto-vectorization implementation of math.rsqrt
add_custom_command(OUTPUT rsqrt_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MathRsqrtOp/MathRsqrt.mlir |
          sed 's/@rsqrt/@rsqrt_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o rsqrt_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} rsqrt_auto_vectorization.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/rsqrt_auto_vectorization.o
)
add_library(MathRsqrtAutoVectorization STATIC rsqrt_auto_vectorization.o)
set_target_properties(MathRsqrtAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

# Add the executable for the benchmark
add_executable(dl-op-linalg-mathrsqrt-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-mathrsqrt-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

# Define the benchmark tool
set(BenchmarkTool GoogleBenchmark)

# Link the benchmark executable with the necessary libraries
target_link_libraries(dl-op-linalg-mathrsqrt-benchmark
  ${BenchmarkTool}
  MathRsqrtScalar
  MathRsqrtAutoVectorization
)
