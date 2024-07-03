add_custom_command(OUTPUT mlir-conv2d-scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2d/Conv2D.mlir |
            sed 's/@conv2d/@conv2d_scalar/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt 
            -conv-vectorization 
            -convert-linalg-to-loops 
            -lower-affine 
            -arith-bufferize 
            -convert-scf-to-cf 
            -convert-vector-to-llvm 
            -convert-arith-to-llvm 
            -finalize-memref-to-llvm 
            -llvm-request-c-wrappers
            -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
            ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2d/mlir-conv2d-scalar.o
)
add_library(Conv2DScalar STATIC mlir-conv2d-scalar.o)
set_target_properties(Conv2DScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT mlir-conv2d-rvv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2d/Conv2D.mlir |
            sed 's/@conv2d/@conv2d_rvv/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
            -conv-vectorization 
            -convert-linalg-to-loops 
            -lower-affine 
            -arith-bufferize 
            -convert-scf-to-cf 
            -convert-vector-to-llvm 
            -convert-arith-to-llvm 
            -finalize-memref-to-llvm 
            -llvm-request-c-wrappers
            -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
            ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
        -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2d/mlir-conv2d-rvv.o
)
add_library(Conv2DRVV STATIC mlir-conv2d-rvv.o)
set_target_properties(Conv2DRVV PROPERTIES LINKER_LANGUAGE CXX)

add_executable(conv2d-benchmark
  Main.cpp
  Conv2DBenchmark.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_link_libraries(conv2d-benchmark
  GoogleBenchmark
  Conv2DScalar
  Conv2DRVV
  )