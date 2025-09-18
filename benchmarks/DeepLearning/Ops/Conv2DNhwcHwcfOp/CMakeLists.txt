separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT conv2d_nhwc_hwcf_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcHwcfOp/Conv2DNhwcHwcf.mlir |
          sed 's/@conv_2d_nhwc_hwcf/@conv_2d_nhwc_hwcf_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o conv2d_nhwc_hwcf_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} conv2d_nhwc_hwcf_scalar.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/conv2d_nhwc_hwcf_scalar.o
)
add_library(Conv2DNhwcHwcfScalar STATIC conv2d_nhwc_hwcf_scalar.o)
set_target_properties(Conv2DNhwcHwcfScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_nhwc_hwcf_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcHwcfOp/Conv2DNhwcHwcf.mlir |
          sed 's/@conv_2d_nhwc_hwcf/@conv_2d_nhwc_hwcf_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o conv2d_nhwc_hwcf_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} conv2d_nhwc_hwcf_auto_vectorization.ll
            -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/conv2d_nhwc_hwcf_auto_vectorization.o
)
add_library(Conv2DNhwcHwcfAutoVectorization STATIC conv2d_nhwc_hwcf_auto_vectorization.o)
set_target_properties(Conv2DNhwcHwcfAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-conv2d-nhwc-hwcf-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-conv2d-nhwc-hwcf-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-conv2d-nhwc-hwcf-benchmark
  ${BenchmarkTool}
  Conv2DNhwcHwcfScalar
  Conv2DNhwcHwcfAutoVectorization
)
