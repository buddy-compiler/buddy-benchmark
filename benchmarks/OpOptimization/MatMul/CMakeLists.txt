add_custom_command(OUTPUT mlir-gemm.o
  COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
  ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/MatMul/gemm.mlir
            --matmul-optimize="vec-size=16;kernel-m=1;kernel-n=2"
            --lower-affine
            -convert-vector-to-llvm
            -convert-memref-to-llvm
            -convert-scf-to-cf
            -convert-linalg-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/MatMul/mlir-gemm.o
)
add_library(MLIRGEMM STATIC mlir-gemm.o)
set_target_properties(MLIRGEMM PROPERTIES LINKER_LANGUAGE CXX)

add_executable(gemm-benchmark
  Main.cpp
  MLIROptBenchmark.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")

target_link_libraries(gemm-benchmark
  GoogleBenchmark
  MLIRGEMM
  ${OpenCV_LIBS}
  )
