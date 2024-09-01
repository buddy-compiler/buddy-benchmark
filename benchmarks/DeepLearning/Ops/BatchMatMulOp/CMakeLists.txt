# BatchMatMul Scalar
add_custom_command(OUTPUT batch_matmul_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMul.mlir |
          sed 's/@batch_matmul/@batch_matmul_scalar/' |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/BatchMatMulOp/batch_matmul_scalar.o
)
add_library(BatchMatMulScalar STATIC batch_matmul_scalar.o)
set_target_properties(BatchMatMulScalar PROPERTIES LINKER_LANGUAGE CXX)

# BatchMatMul Auto Vectorization
add_custom_command(OUTPUT batch_matmul_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMul.mlir |
          sed 's/@batch_matmul/@batch_matmul_auto_vectorization/' |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/BatchMatMulOp/batch_matmul_auto_vectorization.o
)
add_library(BatchMatMulAutoVectorization STATIC batch_matmul_auto_vectorization.o)
set_target_properties(BatchMatMulAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

# BatchMatMul Vectorization
add_custom_command(OUTPUT batch_matmul_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMulVec.mlir |
          sed 's/@batch_matmul/@batch_matmul_vectorization/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/BatchMatMulOp/batch_matmul_vectorization.o
)
add_library(BatchMatMulVectorization STATIC batch_matmul_vectorization.o)
set_target_properties(BatchMatMulVectorization PROPERTIES LINKER_LANGUAGE CXX)

# BatchMatMul Tiling Vectorization
add_custom_command(OUTPUT batch_matmul_vectorization_tile.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMulVecTile.mlir |
          sed 's/@batch_matmul/@batch_matmul_tilling_vectorization/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          --expand-strided-metadata
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts|
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/BatchMatMulOp/batch_matmul_vectorization_tile.o
)
add_library(BatchMatMulTilingVectorization STATIC batch_matmul_vectorization_tile.o)
set_target_properties(BatchMatMulTilingVectorization PROPERTIES LINKER_LANGUAGE CXX)

# Add executable for benchmarking
add_executable(dl-op-linalg-batch-matmul-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-batch-matmul-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-batch-matmul-benchmark
  ${BenchmarkTool}
  BatchMatMulScalar
  BatchMatMulAutoVectorization
  BatchMatMulVectorization
  BatchMatMulTilingVectorization
)
