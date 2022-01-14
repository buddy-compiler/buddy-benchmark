add_buddy_model_benchmark(resnet_v2_50-benchmark
  OpenCV
  MLIR ResNet-V2-50.mlir
  BITCODE resnet-default.o
  LIBRARY ResNetDefault
  OPTIONS
    --linalg-bufferize 
    --convert-linalg-to-loops 
    --convert-vector-to-scf 
    --convert-scf-to-std 
    --func-bufferize 
    --tensor-constant-bufferize 
    --tensor-bufferize 
    --std-bufferize 
    --finalizing-bufferize 
    --lower-affine 
    --convert-vector-to-llvm 
    -std-expand -arith-expand 
    --convert-memref-to-llvm 
    --convert-math-to-llvm 
    --convert-std-to-llvm='emit-c-wrappers=1'
    --reconcile-unrealized-casts
  SOURCE Main.cpp ResNetBenchmark.cpp
)
