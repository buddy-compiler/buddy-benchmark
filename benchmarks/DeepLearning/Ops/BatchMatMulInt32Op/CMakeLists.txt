add_custom_command(OUTPUT batch_matmul_scalar_rvv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/BatchMatMul.mlir |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/batch_matmul_scalar_rvv.o
)
add_library(BatchMatMulScalarRVV STATIC batch_matmul_scalar_rvv.o)
set_target_properties(BatchMatMulScalarRVV PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT batch_matmul_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/BatchMatMul.mlir |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/batch_matmul_auto_vectorization.o
)
add_library(BatchMatMulAutoVectorizationRVV STATIC batch_matmul_auto_vectorization.o)
set_target_properties(BatchMatMulAutoVectorizationRVV PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT batch_matmul_rvv_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/BatchMatMulVectorizationRVV.mlir |
          sed 's/@batch_matmul/@batch_matmul_rvv_vectorization/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -lower-vector-exp 
          -lower-rvv 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/batch_matmul_rvv_vectorization.o
)
add_library(BatchMatMulRVVVectorization STATIC batch_matmul_rvv_vectorization.o)
set_target_properties(BatchMatMulRVVVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT batch_matmul_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/BatchMatMulVectorization.mlir |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/batch_matmul_vectorization.o
)
add_library(BatchMatMuManualVectorization STATIC batch_matmul_vectorization.o)
set_target_properties(BatchMatMuManualVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-batch-matmul-benchmark-rvv
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-batch-matmul-benchmark-rvv PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-batch-matmul-benchmark-rvv
  ${BenchmarkTool}
  BatchMatMulScalarRVV
  BatchMatMulAutoVectorizationRVV
  BatchMatMulRVVVectorization
  BatchMatMuManualVectorization
)
