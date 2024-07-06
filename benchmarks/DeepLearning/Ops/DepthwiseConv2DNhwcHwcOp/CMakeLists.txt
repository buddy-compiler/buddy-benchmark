if(CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

add_custom_command(OUTPUT depthwise_conv_2d_nhwc_hwc_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwc.mlir |
  sed 's/@depthwise_conv_2d_nhwc_hwc/@depthwise_conv_2d_nhwc_hwc_scalar/' |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  -convert-linalg-to-loops
  -convert-scf-to-cf
  -lower-affine
  --finalize-memref-to-llvm
  --llvm-request-c-wrappers
  -convert-func-to-llvm
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O0 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/depthwise_conv_2d_nhwc_hwc_scalar.o
)
add_library(DepthwiseConv2DNhwcHwcScalar STATIC depthwise_conv_2d_nhwc_hwc_scalar.o)
set_target_properties(DepthwiseConv2DNhwcHwcScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT depthwise-conv-2d-nhwc-hwc_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwc.mlir |
  sed 's/@depthwise_conv_2d_nhwc_hwc/@depthwise_conv_2d_nhwc_hwc_auto_vectorization/' |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  -convert-linalg-to-loops
  -convert-scf-to-cf
  -lower-affine
  --finalize-memref-to-llvm
  --llvm-request-c-wrappers
  -convert-func-to-llvm
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/depthwise-conv-2d-nhwc-hwc_auto_vectorization.o
)
add_library(DepthwiseConv2DNhwcHwcAutoVectorization STATIC depthwise-conv-2d-nhwc-hwc_auto_vectorization.o)
set_target_properties(DepthwiseConv2DNhwcHwcAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark
  ${BenchmarkTool}
  DepthwiseConv2DNhwcHwcScalar
  DepthwiseConv2DNhwcHwcAutoVectorization
)
