if (CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O0 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/PoolingNhwcSumOp/pooling_nhwc_sum_scalar.o
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/PoolingNhwcSumOp/pooling_nhwc_sum_auto_vectorization.o
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
