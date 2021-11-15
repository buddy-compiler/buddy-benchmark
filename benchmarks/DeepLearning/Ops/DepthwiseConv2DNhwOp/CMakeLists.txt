set(BUDDY_CONV_OPT_ATTR avx512f)
set(LLVM_MLIR_BINARY_DIR ${BUDDY_CONV_OPT_BUILD_DIR}/../llvm/build/bin)

add_custom_command(OUTPUT depthwise-conv2d-nhw.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwOp/DepthwiseConv2DNhw.mlir 
              -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -lower-affine -convert-scf-to-std --convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts| 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_CONV_OPT_ATTR} 
              --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/DepthwiseConv2DNhwOp/depthwise-conv2d-nhw.o
)

add_library(DepthwiseConv2DNhw depthwise-conv2d-nhw.o)

set_target_properties(DepthwiseConv2DNhw PROPERTIES LINKER_LANGUAGE CXX)

add_executable(depthwise-conv2d-nhw Main.cpp MLIROptBenchmark.cpp)

target_link_libraries(depthwise-conv2d-nhw
  GoogleBenchmark
  DepthwiseConv2DNhw
  Container
  )
