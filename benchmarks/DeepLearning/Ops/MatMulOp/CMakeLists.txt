add_custom_command(OUTPUT matmul-scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/MatMul.mlir |
          sed 's/bm_matmul/matmul_scalar/' |
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/MatMulOp/matmul-scalar.o
)
add_library(MatMulScalar STATIC matmul-scalar.o)
set_target_properties(MatMulScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT matmul-transform.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/MatMulTransform.mlir |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -llvm-request-c-wrappers
            -test-lower-to-llvm |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/MatMulOp/matmul-transform.o
)
add_library(MatMulTransform STATIC matmul-transform.o)
set_target_properties(MatMulTransform PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-matmul-benchmark
  GoogleBenchmarkMain.cpp
  )

set_target_properties(dl-op-linalg-matmul-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-matmul-benchmark
  ${BenchmarkTool}
  MatMulScalar
  MatMulTransform
  )
