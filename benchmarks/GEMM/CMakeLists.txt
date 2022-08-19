add_custom_command(OUTPUT mlir-gemm.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  ${BUDDY_SOURCE_DIR}/benchmarks/GEMM/opt_gemm.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/GEMM/mlir-gemm.o
)
add_library(MLIRGEMM STATIC mlir-gemm.o)
set_target_properties(MLIRGEMM PROPERTIES LINKER_LANGUAGE CXX)

add_executable(gemm-benchmark
  Main.cpp
  MLIROptBenchmark.cpp
  )

target_link_libraries(gemm-benchmark
  GoogleBenchmark
  MLIRGEMM
  ${OpenCV_LIBS}
  )
