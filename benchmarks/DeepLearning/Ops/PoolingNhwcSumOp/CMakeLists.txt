add_buddy_ops_benchmark(pooling-nhwc-sum-benchmark
  MLIR PoolingNhwcSum.mlir
  BITCODE pooling-nhwc-sum.o
  LIBRARY PoolingNhwcSum
  OPTIONS
    -convert-linalg-to-loops 
    -convert-scf-to-std 
    -convert-linalg-to-llvm 
    -lower-affine 
    --convert-memref-to-llvm 
    -convert-std-to-llvm='emit-c-wrappers=1' 
    -reconcile-unrealized-casts
  SOURCE Main.cpp MLIROptBenchmark.cpp
)
