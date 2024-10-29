# Add the executable for the MatMul benchmark
add_executable(dl-op-linalg-matmul-benchmark-rvv
  Main.cpp
)

# Link the Google Benchmark library
target_link_libraries(dl-op-linalg-matmul-benchmark-rvv GoogleBenchmark)

# Separate the compiler flags
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# -----------------------------------------------------------------------------
# Build and link matmul_scalar.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT matmul_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulInt32Op/MatMul.mlir |
          sed 's/@matmul/@matmul_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_scalar.ll
          -c -save-temps -o matmul_scalar.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulInt32Op/MatMul.mlir
)
add_library(matmul_scalar STATIC matmul_scalar.o)
set_target_properties(matmul_scalar PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-benchmark-rvv
  matmul_scalar
)

# -----------------------------------------------------------------------------
# Build and link matmul_vec.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT matmul_vec.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulInt32Op/MatmulVec.mlir |
          sed 's/@matmul/@matmul_vec/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -llvm-request-c-wrappers
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_vec.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_vec.ll
          -c -save-temps -o matmul_vec.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulInt32Op/MatmulVec.mlir
)
add_library(matmul_vec STATIC matmul_vec.o)
set_target_properties(matmul_vec PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-benchmark-rvv
  matmul_vec
)

# -----------------------------------------------------------------------------
# Build and link matmul_rvv.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT matmul_rvv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulInt32Op/MatMulRVV.mlir |
          sed 's/@matmul/@matmul_rvv/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -llvm-request-c-wrappers
            -lower-affine
            -convert-scf-to-cf
            -convert-math-to-llvm
            -lower-vector-exp
            -lower-rvv
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o matmul_rvv.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_rvv.ll
          -c -save-temps -o matmul_rvv.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulInt32Op/MatMulRVV.mlir
)
add_library(matmul_rvv STATIC matmul_rvv.o)
set_target_properties(matmul_rvv PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-benchmark-rvv
  matmul_rvv
)
