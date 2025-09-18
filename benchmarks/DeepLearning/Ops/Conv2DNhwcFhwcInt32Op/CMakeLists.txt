# Add the executable for the Conv2D NHWC-FHWC benchmark
add_executable(dl-op-linalg-conv2d-nhwc-fhwc-benchmark-rvv
  Main.cpp
)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark-rvv GoogleBenchmark)

# CMAKE_C_FLAGS is set when configuring CMake
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# -----------------------------------------------------------------------------
# Build and link conv2d_nhwc_fhwc_scalar.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT conv2d_nhwc_fhwc_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcInt32Op/Conv2DNhwcFhwc.mlir |
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
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcInt32Op/Conv2DNhwcFhwc.mlir
)
add_library(conv2d_nhwc_fhwc_scalar_rvv STATIC conv2d_nhwc_fhwc_scalar.o)
set_target_properties(conv2d_nhwc_fhwc_scalar_rvv PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark-rvv
  conv2d_nhwc_fhwc_scalar_rvv
)

# -----------------------------------------------------------------------------
# Build and link conv2d_nhwc_fhwc_auto_vectorization.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT conv2d_nhwc_fhwc_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcInt32Op/Conv2DNhwcFhwc.mlir |
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
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcInt32Op/Conv2DNhwcFhwc.mlir
)
add_library(conv2d_nhwc_fhwc_rvv_auto_vectorization STATIC conv2d_nhwc_fhwc_auto_vectorization.o)
set_target_properties(conv2d_nhwc_fhwc_rvv_auto_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark-rvv
  conv2d_nhwc_fhwc_rvv_auto_vectorization
)

# -----------------------------------------------------------------------------
# Build and link conv2d_nhwc_fhwc_vectorization.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT conv2d_nhwc_fhwc_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcInt32Op/Conv2DNhwcFhwcVec.mlir |
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
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcInt32Op/Conv2DNhwcFhwcVec.mlir
)
add_library(conv2d_nhwc_fhwc_boardcast_vectorization STATIC conv2d_nhwc_fhwc_vectorization.o)
set_target_properties(conv2d_nhwc_fhwc_boardcast_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark-rvv
conv2d_nhwc_fhwc_boardcast_vectorization
)

# -----------------------------------------------------------------------------
# Build and link conv2d_nhwc_fhwc_rvv_vectorization.o
# -----------------------------------------------------------------------------
add_custom_command(OUTPUT conv2d_nhwc_fhwc_rvv_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcInt32Op/Conv2DNhwcFhwcVecRVV.mlir |
          sed 's/@conv_2d_nhwc_fhwc/@conv_2d_nhwc_fhwc_rvv_vectorization/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -lower-affine
            -convert-scf-to-cf
            -convert-math-to-llvm
            -lower-vector-exp
            -lower-rvv
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
            ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o conv2d_nhwc_fhwc_rvv_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} conv2d_nhwc_fhwc_rvv_vectorization.ll
          -c -save-temps -o conv2d_nhwc_fhwc_rvv_vectorization.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNhwcFhwcInt32Op/Conv2DNhwcFhwcVecRVV.mlir
)
add_library(conv2d_nhwc_fhwc_rvv_vectorization STATIC conv2d_nhwc_fhwc_rvv_vectorization.o)
set_target_properties(conv2d_nhwc_fhwc_rvv_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-conv2d-nhwc-fhwc-benchmark-rvv
  conv2d_nhwc_fhwc_rvv_vectorization
)
