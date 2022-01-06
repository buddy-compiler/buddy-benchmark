set(LLVM_MLIR_BINARY_DIR ${BUDDY_OPT_BUILD_DIR}/../llvm/build/bin)

add_custom_command(OUTPUT conv-2d-nchw-fchw.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/Conv2DNchwFchwOp/Conv2DNchwFchw.mlir 
              -convert-linalg-to-loops 
              -convert-scf-to-std 
              -convert-linalg-to-llvm 
              -lower-affine 
              --convert-memref-to-llvm 
              -convert-std-to-llvm='emit-c-wrappers=1' 
              -reconcile-unrealized-casts| 
                ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
                ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_OPT_ATTR} 
              --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/Conv2DNchwFchwOp/conv-2d-nchw-fchw.o
)

add_library(Conv2DNchwFchw conv-2d-nchw-fchw.o)

set_target_properties(Conv2DNchwFchw PROPERTIES LINKER_LANGUAGE CXX)

add_executable(conv-2d-nchw-fchw-benchmark Main.cpp MLIROptBenchmark.cpp)

target_link_libraries(conv-2d-nchw-fchw-benchmark
  GoogleBenchmark
  Conv2DNchwFchw
  Container
)
