# Add the executable for the Depthwise Conv2D NHWC-HWC benchmark
add_executable(dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark
  Main.cpp
)

# Link the Google Benchmark library
target_link_libraries(dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark GoogleBenchmark)

# Separate the compiler flags
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# -----------------------------------------------------------------------------
# Build and link depthwise_conv_2d_nhwc_hwc_scalar.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT depthwise_conv_2d_nhwc_hwc_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwc.mlir |
          sed 's/@depthwise_conv_2d_nhwc_hwc/@depthwise_conv_2d_nhwc_hwc_scalar/' |
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o depthwise_conv_2d_nhwc_hwc_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} depthwise_conv_2d_nhwc_hwc_scalar.ll
          -c -save-temps -o depthwise_conv_2d_nhwc_hwc_scalar.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwc.mlir
)
add_library(depthwise_conv_2d_nhwc_hwc_scalar STATIC depthwise_conv_2d_nhwc_hwc_scalar.o)
set_target_properties(depthwise_conv_2d_nhwc_hwc_scalar PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark
  depthwise_conv_2d_nhwc_hwc_scalar
)

# -----------------------------------------------------------------------------
# Build and link depthwise_conv_2d_nhwc_hwc_auto_vectorization.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT depthwise_conv_2d_nhwc_hwc_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwc.mlir |
          sed 's/@depthwise_conv_2d_nhwc_hwc/@depthwise_conv_2d_nhwc_hwc_auto_vectorization/' |
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o depthwise_conv_2d_nhwc_hwc_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} depthwise_conv_2d_nhwc_hwc_auto_vectorization.ll
          -c -save-temps -o depthwise_conv_2d_nhwc_hwc_auto_vectorization.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwc.mlir
)
add_library(depthwise_conv_2d_nhwc_hwc_auto_vectorization STATIC depthwise_conv_2d_nhwc_hwc_auto_vectorization.o)
set_target_properties(depthwise_conv_2d_nhwc_hwc_auto_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark
  depthwise_conv_2d_nhwc_hwc_auto_vectorization
)

# -----------------------------------------------------------------------------
# Build and link depthwise_conv_2d_nhwc_hwc_vectorization.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT depthwise_conv_2d_nhwc_hwc_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwcVec.mlir |
          sed 's/@depthwise_conv_2d_nhwc_hwc/@depthwise_conv_2d_nhwc_hwc_vectorization/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -lower-affine
            -convert-scf-to-cf
            -convert-math-to-llvm
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o depthwise_conv_2d_nhwc_hwc_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} depthwise_conv_2d_nhwc_hwc_vectorization.ll
          -c -save-temps -o depthwise_conv_2d_nhwc_hwc_vectorization.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwcHwcOp/DepthwiseConv2DNhwcHwcVec.mlir
)
add_library(depthwise_conv_2d_nhwc_hwc_vectorization STATIC depthwise_conv_2d_nhwc_hwc_vectorization.o)
set_target_properties(depthwise_conv_2d_nhwc_hwc_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark
  depthwise_conv_2d_nhwc_hwc_vectorization
)
