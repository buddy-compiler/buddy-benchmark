add_buddy_ops_benchmark(gemm-benchmark
  MLIR opt_gemm.mlir
  BITCODE gemm.o
  LIBRARY GEMM
  OPTIONS
    --lower-affine

    -convert-scf-to-cf 
    --convert-memref-to-llvm
    --llvm-request-c-wrappers
    -convert-func-to-llvm 
    -reconcile-unrealized-casts
  SOURCE Main.cpp MLIROptBenchmark.cpp
)
