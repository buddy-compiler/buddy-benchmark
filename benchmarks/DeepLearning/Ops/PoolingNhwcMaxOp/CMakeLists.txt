add_custom_command(OUTPUT pooling_nhwc_max-scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/PoolingNhwcMaxOp/PoolingNhwcMax.mlir |
  sed 's/@pooling_nhwc_max/@pooling_nhwc_max_scalar/' |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  -convert-linalg-to-loops 
  -convert-scf-to-cf 
  -lower-affine
  -convert-vector-to-llvm  
  -finalize-memref-to-llvm 
  --llvm-request-c-wrappers
  -convert-func-to-llvm 
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/PoolingNhwcMaxOp/pooling_nhwc_max-scalar.o
)
add_library(PoolingNhwcMaxScalar STATIC pooling_nhwc_max-scalar.o)
set_target_properties(PoolingNhwcMaxScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT pooling_nhwc_max-vec.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/PoolingNhwcMaxOp/PoolingNhwcMax.mlir |
  sed 's/@pooling_nhwc_max/@pooling_nhwc_max_vec/' |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  -pooling-nhwc-max-vectorization
  -convert-linalg-to-loops 
  -convert-vector-to-scf
  -lower-affine 
  -arith-bufferize 
  -convert-scf-to-cf 
  -convert-vector-to-llvm 
  -convert-arith-to-llvm 
  -finalize-memref-to-llvm 
  -llvm-request-c-wrappers
  -convert-func-to-llvm 
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/PoolingNhwcMaxOp/pooling_nhwc_max-vec.o
)
add_library(PoolingNhwcMaxVectorization STATIC pooling_nhwc_max-vec.o)
set_target_properties(PoolingNhwcMaxVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT pooling_nhwc_max-adapt-vec.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/PoolingNhwcMaxOp/PoolingNhwcMax.mlir |
  sed 's/@pooling_nhwc_max/@pooling_nhwc_max_adapt_vec/' |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  -pooling-nhwc-max-adapt-vectorization
  -convert-linalg-to-loops 
  -convert-vector-to-scf
  -lower-affine 
  -arith-bufferize 
  -convert-scf-to-cf 
  -convert-vector-to-llvm 
  -convert-arith-to-llvm 
  -finalize-memref-to-llvm 
  -llvm-request-c-wrappers
  -convert-func-to-llvm 
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/PoolingNhwcMaxOp/pooling_nhwc_max-adapt-vec.o
)
add_library(PoolingNhwcMaxAdaptVectorization STATIC pooling_nhwc_max-adapt-vec.o)
set_target_properties(PoolingNhwcMaxAdaptVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-pooling-nhwc-max-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-pooling-nhwc-max-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-pooling-nhwc-max-benchmark
  ${BenchmarkTool}
  PoolingNhwcMaxScalar
  PoolingNhwcMaxVectorization
  PoolingNhwcMaxAdaptVectorization
)
