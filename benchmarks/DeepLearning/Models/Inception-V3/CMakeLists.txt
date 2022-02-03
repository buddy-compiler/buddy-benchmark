add_buddy_model_benchmark(inception-v3-benchmark
  OpenCV
  MLIR Inception.mlir
  BITCODE inception-v3-default.o
  LIBRARY InceptionV3Default
  OPTIONS
    --linalg-bufferize 
    --convert-linalg-to-loops 
    --convert-vector-to-scf 
    --convert-scf-to-std 
    --func-bufferize 
    --arith-bufferize 
    --tensor-bufferize 
    --finalizing-bufferize 
    --lower-affine 
    --convert-vector-to-llvm 
    -memref-expand -arith-expand 
    --convert-memref-to-llvm 
    --convert-math-to-llvm 
    --convert-std-to-llvm='emit-c-wrappers=1'
    --reconcile-unrealized-casts
  SOURCE Main.cpp InceptionBenchmark.cpp
)
