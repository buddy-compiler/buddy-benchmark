add_buddy_model_benchmark(squeezenet-benchmark
  OpenCV
  MLIR Squeezenet.mlir
  BITCODE squeezenet-default.o
  LIBRARY SqueezenetDefault
  OPTIONS
    --split-input-file
    --tosa-to-linalg-named
    --tosa-to-linalg
    --tosa-to-standard
    --linalg-bufferize 
    --convert-linalg-to-loops 
    --convert-vector-to-scf 
    --convert-scf-to-cf 
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
  SOURCE Main.cpp SqueezenetBenchmark.cpp
)
