cmake_minimum_required(VERSION 3.10)
project(ArithAddfBenchmark)

# Define variables for the cross-compilation toolchain and options.
if (CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

# Add custom commands for scalar implementation of arith.addf
add_custom_command(OUTPUT addf_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/ArithAddfOp/ArithAddf.mlir |
          sed 's/@addf/@addf_scalar/' |
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/ArithAddfOp/addf_scalar.o
)
add_library(ArithAddfScalar STATIC addf_scalar.o)
set_target_properties(ArithAddfScalar PROPERTIES LINKER_LANGUAGE CXX)

# Add custom commands for auto-vectorization implementation of arith.addf
add_custom_command(OUTPUT addf_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/ArithAddfOp/ArithAddf.mlir |
          sed 's/@addf/@addf_auto_vectorization/' |
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/ArithAddfOp/addf_auto_vectorization.o
)
add_library(ArithAddfAutoVectorization STATIC addf_auto_vectorization.o)
set_target_properties(ArithAddfAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

# Add the executable for the benchmark
add_executable(dl-op-linalg-arithaddf-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-arithaddf-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

# Define the benchmark tool
set(BenchmarkTool GoogleBenchmark)

# Link the benchmark executable with the necessary libraries
target_link_libraries(dl-op-linalg-arithaddf-benchmark
  ${BenchmarkTool}
  ArithAddfScalar
  ArithAddfAutoVectorization
)
