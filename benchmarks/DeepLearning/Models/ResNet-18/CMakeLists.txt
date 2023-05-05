set(RESNET18_TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-tensor),func.func(tosa-to-linalg),func.func(tosa-to-arith))")

# Compile MLIR file to object file.
add_custom_command(OUTPUT resnet-18-default.o
COMMAND 
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/ResNet-18.mlir
    --pass-pipeline="${RESNET18_TOSA_PIPELINE}" | 
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    --test-linalg-transform-patterns="test-generalize-pad-tensor"
    --linalg-bufferize
    --convert-linalg-to-loops
    --func-bufferize
    --arith-bufferize
    --tensor-bufferize 
    --finalizing-bufferize
    --convert-vector-to-scf
    --convert-scf-to-cf
    --expand-strided-metadata
    --lower-affine
    --convert-vector-to-llvm
    --memref-expand
    --arith-expand
    --convert-arith-to-llvm
    --finalize-memref-to-llvm
    --convert-math-to-llvm
    --llvm-request-c-wrappers
    --convert-func-to-llvm
    --reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} 
    --filetype=obj -o ${CMAKE_CURRENT_BINARY_DIR}/resnet-18-default.o
)

add_library(ResNet18Default STATIC resnet-18-default.o)
set_target_properties(ResNet18Default PROPERTIES LINKER_LANGUAGE CXX)

add_executable(resnet-18-benchmark Main.cpp ResNet18DefaultBenchmark.cpp)
# Link libraries
target_link_directories(resnet-18-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(resnet-18-benchmark
  ResNet18Default
  GoogleBenchmark
  mlir_c_runner_utils
  ${OpenCV_LIBS}
)
