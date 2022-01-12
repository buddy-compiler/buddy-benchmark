set(LLVM_MLIR_BINARY_DIR ${BUDDY_OPT_BUILD_DIR}/../llvm/build/bin)

add_custom_command(OUTPUT resnet-default.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Models/ResNet-V2-50/ResNet.mlir 
              --linalg-bufferize --convert-linalg-to-loops --convert-vector-to-scf --convert-scf-to-std 
              --func-bufferize --tensor-constant-bufferize --tensor-bufferize --std-bufferize --finalizing-bufferize 
              --lower-affine --convert-vector-to-llvm -std-expand -arith-expand --convert-memref-to-llvm --convert-math-to-llvm 
              --convert-std-to-llvm='emit-c-wrappers=1' --reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_OPT_ATTR} 
              --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Models/ResNet-V2-50/resnet-default.o
)

add_library(ResNetDefault resnet-default.o)

set_target_properties(ResNetDefault PROPERTIES LINKER_LANGUAGE CXX)

add_executable(resnet-benchmark Main.cpp ResNetBenchmark.cpp)

target_link_libraries(resnet-benchmark
  GoogleBenchmark
  ${OpenCV_LIBS}
  ResNetDefault
  Container
  PNGImage
  )
