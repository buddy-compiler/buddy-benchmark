add_custom_command(OUTPUT conv2d_nhwc_fhwc_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir |
  sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_scalar/' |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  -convert-linalg-to-loops 
  -convert-scf-to-cf 
  -lower-affine
  -arith-bufferize 
  -convert-vector-to-llvm  
  -finalize-memref-to-llvm 
  -convert-arith-to-llvm
  --llvm-request-c-wrappers
  -convert-func-to-llvm 
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O0 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/conv2d_nhwc_fhwc_scalar.o
)
add_library(Conv2DNhwcFhwcScalar STATIC conv2d_nhwc_fhwc_scalar.o)
set_target_properties(Conv2DNhwcFhwcScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_nhwc_fhwc_auto_vec.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir |
  sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_auto_vec/' |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  -convert-linalg-to-loops 
  -convert-scf-to-cf 
  -lower-affine
  -arith-bufferize 
  -convert-vector-to-llvm  
  -finalize-memref-to-llvm 
  -convert-arith-to-llvm
  --llvm-request-c-wrappers
  -convert-func-to-llvm 
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/conv2d_nhwc_fhwc_auto_vec.o
)
add_library(Conv2DNhwcFhwcAutoVectorization STATIC conv2d_nhwc_fhwc_auto_vec.o)
set_target_properties(Conv2DNhwcFhwcAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_nhwc_fhwc_vec.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir |
  sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_vec/' |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt 
  -conv2d-nhwc-fhwc-vectorization 
  -convert-vector-to-scf 
  -lower-affine 
  -arith-bufferize 
  -convert-scf-to-cf 
  -convert-vector-to-llvm 
  -convert-arith-to-llvm 
  -finalize-memref-to-llvm 
  --llvm-request-c-wrappers 
  -convert-func-to-llvm 
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir | 
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/conv2d_nhwc_fhwc_vec.o 
)
add_library(Conv2DNhwcFhwcVectorization STATIC conv2d_nhwc_fhwc_vec.o)
set_target_properties(Conv2DNhwcFhwcVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_nhwc_fhwc_adapt.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir |
  sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_adapt/' |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt 
  -conv2d-nhwc-fhwc-adapt-vectorization
  -convert-vector-to-scf 
  -lower-affine 
  -arith-bufferize 
  -convert-scf-to-cf 
  -convert-vector-to-llvm 
  -convert-arith-to-llvm 
  -finalize-memref-to-llvm 
  --llvm-request-c-wrappers 
  -convert-func-to-llvm 
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir | 
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/conv2d_nhwc_fhwc_adapt.o 
)
add_library(Conv2DNhwcFhwcAdaptVectorization STATIC conv2d_nhwc_fhwc_adapt.o)
set_target_properties(Conv2DNhwcFhwcAdaptVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-conv2d-nhwc-fhwc-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-conv2d-nhwc-fhwc-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark
  ${BenchmarkTool}
  Conv2DNhwcFhwcScalar
  Conv2DNhwcFhwcAutoVectorization
  Conv2DNhwcFhwcVectorization
  Conv2DNhwcFhwcAdaptVectorization
)
    