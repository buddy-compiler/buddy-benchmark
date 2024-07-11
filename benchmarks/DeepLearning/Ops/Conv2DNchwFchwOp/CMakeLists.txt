if(CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

add_custom_command(OUTPUT conv2d_nchw_fchw-scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNchwFchwOp/Conv2DNchwFchw.mlir |
  sed 's/@conv_2d_nchw_fchw/@conv_2d_nchw_fchw_scalar/' |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  -convert-linalg-to-loops
  -convert-scf-to-cf
  -lower-affine
  -convert-vector-to-llvm
  --finalize-memref-to-llvm
  --llvm-request-c-wrappers
  -convert-func-to-llvm
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNchwFchwOp/conv2d_nchw_fchw-scalar.o
)
add_library(Conv2DNchwFchwScalar STATIC conv2d_nchw_fchw-scalar.o)
set_target_properties(Conv2DNchwFchwScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_nchw_fchw_im2col.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNchwFchwOp/conv2d-nchw-fchw-im2col.mlir |
  sed 's/@conv2d_nchw_fchw_im2col/@conv_2d_nchw_fchw_im2col/' |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  -lower-affine
  -convert-linalg-to-loops
  -convert-vector-to-scf
  -convert-scf-to-cf
  -expand-strided-metadata
  -lower-affine
  -convert-vector-to-llvm
  -memref-expand
  -arith-expand
  -convert-arith-to-llvm
  -finalize-memref-to-llvm
  -convert-math-to-llvm
  --llvm-request-c-wrappers
  -convert-func-to-llvm
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNchwFchwOp/conv2d_nchw_fchw_im2col.o
)
add_library(Conv2DNchwFchwIm2col STATIC conv2d_nchw_fchw_im2col.o)
set_target_properties(Conv2DNchwFchwIm2col PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-conv2d-nchw-fchw-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-conv2d-nchw-fchw-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-conv2d-nchw-fchw-benchmark
  ${BenchmarkTool}
  Conv2DNchwFchwScalar
  Conv2DNchwFchwIm2col
)
