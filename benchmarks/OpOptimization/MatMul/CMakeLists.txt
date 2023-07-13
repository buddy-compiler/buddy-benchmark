add_custom_command(OUTPUT matmul-ocv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/MatMul/MatMul.mlir |
          sed 's/bm_matmul/matmul_ocv/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
            -matmul-optimize="vec-size=16;kernel-m=1;kernel-n=2"
            -expand-strided-metadata
            -lower-affine
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-linalg-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/MatMul/matmul-ocv.o
)
add_library(MatMulOCV STATIC matmul-ocv.o)
set_target_properties(MatMulOCV PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT matmul-scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/MatMul/MatMul.mlir |
          sed 's/bm_matmul/matmul_scalar/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/MatMul/matmul-scalar.o
)
add_library(MatMulScalar STATIC matmul-scalar.o)
set_target_properties(MatMulScalar PROPERTIES LINKER_LANGUAGE CXX)

add_executable(matmul-benchmark
  Main.cpp
  MatMulBenchmark.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_link_libraries(matmul-benchmark
  GoogleBenchmark
  MatMulOCV
  MatMulScalar
  )
