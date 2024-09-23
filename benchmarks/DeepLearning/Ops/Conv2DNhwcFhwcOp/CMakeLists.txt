add_custom_command(OUTPUT conv2d_nhwc_fhwc_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir |
          sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O0 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/conv2d_nhwc_fhwc_scalar.o
)
add_library(Conv2DNhwcFhwcScalar STATIC conv2d_nhwc_fhwc_scalar.o)
set_target_properties(Conv2DNhwcFhwcScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_nhwc_fhwc_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir |
          sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/conv2d_nhwc_fhwc_auto_vectorization.o
)
add_library(Conv2DNhwcFhwcAutoVectorization STATIC conv2d_nhwc_fhwc_auto_vectorization.o)
set_target_properties(Conv2DNhwcFhwcAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_nhwc_fhwc_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwcVec.mlir |
          sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_vectorization/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/conv2d_nhwc_fhwc_vectorization.o
)
add_library(Conv2DNhwcFhwcVectorization STATIC conv2d_nhwc_fhwc_vectorization.o)
set_target_properties(Conv2DNhwcFhwcVectorization PROPERTIES LINKER_LANGUAGE CXX)


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
  )
