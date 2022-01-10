set(LLVM_MLIR_BINARY_DIR ${BUDDY_OPT_BUILD_DIR}/../llvm/build/bin)

add_custom_command(OUTPUT pooling-nhwc-sum.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/PoolingNhwcSumOp/PoolingNhwcSum.mlir 
    -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -lower-affine
    -convert-scf-to-std --convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts| 
     ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
     ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_OPT_ATTR} 
     --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/PoolingNhwcSumOp/pooling-nhwc-sum.o
)

add_library(PoolingNhwcSum pooling-nhwc-sum.o)

set_target_properties(PoolingNhwcSum PROPERTIES LINKER_LANGUAGE CXX)

add_executable(pooling-nhwc-sum-benchmark Main.cpp MLIROptBenchmark.cpp)

target_link_libraries(pooling-nhwc-sum-benchmark
  GoogleBenchmark
  PoolingNhwcSum
  Container
  PNGImage
)
