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

add_custom_command(OUTPUT matmul-transform.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/MatMul/MatMulTransform.mlir |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            --test-transform-dialect-interpreter
            --test-transform-dialect-erase-schedule
            --convert-vector-to-scf
            --convert-scf-to-cf
            --convert-linalg-to-llvm
            --expand-strided-metadata
            --lower-affine
            --convert-vector-to-llvm
            --memref-expand
            --arith-expand
            --convert-arith-to-llvm
            --finalize-memref-to-llvm
            --convert-math-to-llvm
            -llvm-request-c-wrappers
            --convert-func-to-llvm
            --reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/MatMul/matmul-transform.o
)
add_library(MatMulTransform STATIC matmul-transform.o)
set_target_properties(MatMulTransform PROPERTIES LINKER_LANGUAGE CXX)

function(build_matmul_broadcast step)
  add_custom_command(OUTPUT matmul-broadcast-${step}.o
    COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/MatMul/MatMulBroadcast.mlir |
            sed 's/STEP_PLACEHOLDER/${step}/g' |
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
              -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/MatMul/matmul-broadcast-${step}.o
  )
  add_library(MatMulBroadcast${step} STATIC matmul-broadcast-${step}.o)
  set_target_properties(MatMulBroadcast${step} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_matmul_broadcast(16)
build_matmul_broadcast(32)
build_matmul_broadcast(64)
build_matmul_broadcast(128)
build_matmul_broadcast(256)

add_custom_command(OUTPUT matmul-scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/MatMul/MatMul.mlir |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/MatMul/matmul-scalar.o
)
add_library(MatMulScalar STATIC matmul-scalar.o)
set_target_properties(MatMulScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT batch-matmul-scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/MatMul/BatchMatMul.mlir |
          sed 's/bm_batch_matmul/batch_matmul_scalar/' |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/MatMul/batch-matmul-scalar.o
)
add_library(BatchMatMulScalar STATIC batch-matmul-scalar.o)
set_target_properties(BatchMatMulScalar PROPERTIES LINKER_LANGUAGE CXX)

function(build_batch_matmul_broadcast step)
  add_custom_command(OUTPUT batch-matmul-broadcast-${step}.o
    COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/MatMul/BatchMatMul.mlir |
          sed 's/bm_batch_matmul/batch_matmul_broadcast_${step}/g' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
            -batchmatmul-optimize="step-placeholder=${step}"
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/MatMul/batch-matmul-broadcast-${step}.o
  )
  add_library(BatchMatMulBroadcast${step} STATIC batch-matmul-broadcast-${step}.o)
  set_target_properties(BatchMatMulBroadcast${step} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_batch_matmul_broadcast(64)

add_executable(matmul-benchmark
  Main.cpp
  MatMulBenchmark.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_link_libraries(matmul-benchmark
  GoogleBenchmark
  MatMulOCV
  MatMulTransform
  MatMulBroadcast16
  MatMulBroadcast32
  MatMulBroadcast64
  MatMulBroadcast128
  MatMulBroadcast256
  MatMulScalar
  BatchMatMulScalar
  BatchMatMulBroadcast64
  )
