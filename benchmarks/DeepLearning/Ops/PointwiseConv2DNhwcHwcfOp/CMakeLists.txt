set(BUDDY_OPT_ATTR avx512f)
set(LLVM_MLIR_BINARY_DIR ${BUDDY_OPT_BUILD_DIR}/../llvm/build/bin)

add_custom_command(OUTPUT pointwise-conv-2d-nhwc-hwcf.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/PointwiseConv2DNhwcHwcfOp/PointwiseConv2DNhwcHwcf.mlir 
              -linalg-bufferize 
              -std-bufferize 
              -tensor-constant-bufferize 
              -tensor-bufferize 
              -func-bufferize 
              -finalizing-bufferize 
              -buffer-deallocation
              -convert-linalg-to-loops 
              -convert-scf-to-std 
              -convert-linalg-to-llvm 
              -lower-affine 
              --convert-memref-to-llvm 
              -convert-std-to-llvm='emit-c-wrappers=1' 
              -reconcile-unrealized-casts| 
                ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
                ${LLVM_MLIR_BINARY_DIR}/llc 
              -mtriple=x86_64-unknown-linux-gnu 
              -mattr=${BUDDY_OPT_ATTR} 
              --filetype=obj 
              -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/PointwiseConv2DNhwcHwcfOp/pointwise-conv-2d-nhwc-hwcf.o
)

add_library(PointwiseConv2DNhwcHwcf pointwise-conv-2d-nhwc-hwcf.o)

set_target_properties(PointwiseConv2DNhwcHwcf PROPERTIES LINKER_LANGUAGE CXX)

add_executable(pointwise-conv-2d-nhwc-hwcf-benchmark Main.cpp MLIROptBenchmark.cpp)

target_link_libraries(pointwise-conv-2d-nhwc-hwcf-benchmark
  GoogleBenchmark
  PointwiseConv2DNhwcHwcf
  Container
)
