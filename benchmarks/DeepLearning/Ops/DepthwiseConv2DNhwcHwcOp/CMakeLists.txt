add_buddy_ops_benchmark(depthwise-conv-2d-nhwc-hwc-benchmark
  MLIR DepthwiseConv2DNhwcHwc.mlir
  BITCODE depthwise-conv-2d-nhwc-hwc.o
  LIBRARY DepthwiseConv2DNhwcHwc
  OPTIONS
    -convert-linalg-to-loops 
    -convert-scf-to-cf 
    -convert-linalg-to-llvm 
    -lower-affine 
    --convert-memref-to-llvm 
    -convert-func-to-llvm='emit-c-wrappers=1' 
    -reconcile-unrealized-casts
  SOURCE Main.cpp MLIROptBenchmark.cpp
)
