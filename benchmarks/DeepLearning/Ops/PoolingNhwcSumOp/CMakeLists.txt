separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT pooling_nhwc_sum_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/PoolingNhwcSumOp/PoolingNhwcSum.mlir |
          sed 's/@pooling_nhwc_sum/@pooling_nhwc_sum_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o pooling_nhwc_sum_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} pooling_nhwc_sum_scalar.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/pooling_nhwc_sum_scalar.o
)
add_library(PoolingNhwcSumScalar STATIC pooling_nhwc_sum_scalar.o)
set_target_properties(PoolingNhwcSumScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT pooling_nhwc_sum_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/PoolingNhwcSumOp/PoolingNhwcSum.mlir |
          sed 's/@pooling_nhwc_sum/@pooling_nhwc_sum_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o pooling_nhwc_sum_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} pooling_nhwc_sum_auto_vectorization.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/pooling_nhwc_sum_auto_vectorization.o
)
add_library(PoolingNhwcSumAutoVectorization STATIC pooling_nhwc_sum_auto_vectorization.o)
set_target_properties(PoolingNhwcSumAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-pooling-nhwc-sum-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-pooling-nhwc-sum-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-pooling-nhwc-sum-benchmark
  ${BenchmarkTool}
  PoolingNhwcSumScalar
  PoolingNhwcSumAutoVectorization
)
