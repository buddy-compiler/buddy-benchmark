if (CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

add_custom_command(OUTPUT conv2d_nhwc_hwcf_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcHwcfOp/Conv2DNhwcHwcf.mlir |
          sed 's/@conv_2d_nhwc_hwcf/@conv_2d_nhwc_hwcf_scalar/' |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcHwcfOp/conv2d_nhwc_hwcf_scalar.o
)
add_library(Conv2DNhwcHwcfScalar STATIC conv2d_nhwc_hwcf_scalar.o)
set_target_properties(Conv2DNhwcHwcfScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_nhwc_hwcf_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcHwcfOp/Conv2DNhwcHwcf.mlir |
          sed 's/@conv_2d_nhwc_hwcf/@conv_2d_nhwc_hwcf_auto_vectorization/' |
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
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcHwcfOp/conv2d_nhwc_hwcf_auto_vectorization.o
)
add_library(Conv2DNhwcHwcfAutoVectorization STATIC conv2d_nhwc_hwcf_auto_vectorization.o)
set_target_properties(Conv2DNhwcHwcfAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-conv2dnhwchwcf-benchmark
  GoogleBenchmarkMain.cpp
  )

set_target_properties(dl-op-linalg-conv2dnhwchwcf-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-conv2dnhwchwcf-benchmark
  ${BenchmarkTool}
  Conv2DNhwcHwcfScalar
  Conv2DNhwcHwcfAutoVectorization
  )
