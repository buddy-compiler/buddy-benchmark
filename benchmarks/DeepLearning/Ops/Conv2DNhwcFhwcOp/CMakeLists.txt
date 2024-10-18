# Add the executable for the Conv2D NHWC-FHWC benchmark
add_executable(dl-op-linalg-conv2d-nhwc-fhwc-benchmark
  Main.cpp
)

# Link the Google Benchmark library
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark GoogleBenchmark)

# Separate the compiler flags
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# -----------------------------------------------------------------------------
# Build and link conv2d_nhwc_fhwc_scalar.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT conv2d_nhwc_fhwc_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir |
          sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o conv2d_nhwc_fhwc_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} conv2d_nhwc_fhwc_scalar.ll
          -c -save-temps -o conv2d_nhwc_fhwc_scalar.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir
)
add_library(conv2d_nhwc_fhwc_scalar STATIC conv2d_nhwc_fhwc_scalar.o)
set_target_properties(conv2d_nhwc_fhwc_scalar PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark
  conv2d_nhwc_fhwc_scalar
)

# -----------------------------------------------------------------------------
# Build and link conv2d_nhwc_fhwc_auto_vectorization.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT conv2d_nhwc_fhwc_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir |
          sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o conv2d_nhwc_fhwc_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} conv2d_nhwc_fhwc_auto_vectorization.ll
          -c -save-temps -o conv2d_nhwc_fhwc_auto_vectorization.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwc.mlir
)
add_library(conv2d_nhwc_fhwc_auto_vectorization STATIC conv2d_nhwc_fhwc_auto_vectorization.o)
set_target_properties(conv2d_nhwc_fhwc_auto_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark
  conv2d_nhwc_fhwc_auto_vectorization
)

# -----------------------------------------------------------------------------
# Build and link conv2d_nhwc_fhwc_vectorization.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT conv2d_nhwc_fhwc_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwcVec.mlir |
          sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_vectorization/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -lower-affine
            -convert-scf-to-cf
            -convert-math-to-llvm
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o conv2d_nhwc_fhwc_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} conv2d_nhwc_fhwc_vectorization.ll
          -c -save-temps -o conv2d_nhwc_fhwc_vectorization.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwcVec.mlir
)
add_library(conv2d_nhwc_fhwc_vectorization STATIC conv2d_nhwc_fhwc_vectorization.o)
set_target_properties(conv2d_nhwc_fhwc_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark
  conv2d_nhwc_fhwc_vectorization
)

# -----------------------------------------------------------------------------
# Build and link conv2d_nhwc_fhwc_vec_tile.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT conv2d_nhwc_fhwc_vec_tile.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwcVecTile.mlir |
          sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_vec_tile/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -lower-affine
            -convert-scf-to-cf
            -convert-math-to-llvm
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o conv2d_nhwc_fhwc_vec_tile.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} conv2d_nhwc_fhwc_vec_tile.ll
          -c -save-temps -o conv2d_nhwc_fhwc_vec_tile.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcOp/Conv2DNhwcFhwcVecTile.mlir
)
add_library(conv2d_nhwc_fhwc_vec_tile STATIC conv2d_nhwc_fhwc_vec_tile.o)
set_target_properties(conv2d_nhwc_fhwc_vec_tile PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark
  conv2d_nhwc_fhwc_vec_tile
)

set_target_properties(dl-op-linalg-conv2d-nhwc-fhwc-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark
  ${BenchmarkTool}
)