set(BUDDY_CONV_OPT_ATTR avx512f)
set(LLVM_MLIR_BINARY_DIR ${BUDDY_CONV_OPT_BUILD_DIR}/../llvm/build/bin)

add_custom_command(OUTPUT depthwise-conv-2d-nhwc-hwc.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwc.mlir 
              -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -lower-affine -convert-scf-to-std --convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts| 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_CONV_OPT_ATTR} 
              --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/depthwise-conv-2d-nhwc-hwc.o
)

add_library(DepthwiseConv2DNhwcHwc depthwise-conv-2d-nhwc-hwc.o)

set_target_properties(DepthwiseConv2DNhwcHwc PROPERTIES LINKER_LANGUAGE CXX)

add_executable(depthwise-conv-2d-nhwc-hwc-benchmark Main.cpp MLIROptBenchmark.cpp)

target_link_libraries(depthwise-conv-2d-nhwc-hwc-benchmark
  GoogleBenchmark
  DepthwiseConv2DNhwcHwc
  Container
  PNGImage
  )
