set(alexnet_TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))")

# Compile MLIR file to object file.
add_custom_command(OUTPUT alexnet-default.o
COMMAND 
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/alexnet_tosa.mlir
    --pass-pipeline="${alexnet_TOSA_PIPELINE}" | 
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
    --filetype=obj -o ${CMAKE_CURRENT_BINARY_DIR}/alexnet-default.o
)

add_library(alexnetDefault STATIC alexnet-default.o)
set_target_properties(alexnetDefault PROPERTIES LINKER_LANGUAGE CXX)

add_executable(alexnet-benchmark Main.cpp AlexNetDefaultBenchmark.cpp)
# Link libraries
target_link_directories(alexnet-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(alexnet-benchmark
  alexnetDefault
  GoogleBenchmark
  mlir_c_runner_utils
  ${OpenCV_LIBS}
)
