set(QUANT_TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor))")

# Compile MLIR file to object file.
add_custom_command(OUTPUT efficientnet-quant.o
COMMAND 
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/efficientnet.mlir
    --pass-pipeline="${QUANT_TOSA_PIPELINE}" | 
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    --tosa-to-arith="include-apply-rescale=1"
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
    --filetype=obj -o ${CMAKE_CURRENT_BINARY_DIR}/efficientnet-quant.o
)

add_library(EfficientNetQuant STATIC efficientnet-quant.o)
set_target_properties(EfficientNetQuant PROPERTIES LINKER_LANGUAGE CXX)

add_executable(efficientnet-quant-benchmark Main.cpp EfficientNetQuantBenchmark.cpp)
# Link libraries
target_link_directories(efficientnet-quant-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(efficientnet-quant-benchmark
  EfficientNetQuant
  GoogleBenchmark
  mlir_c_runner_utils
  ${OpenCV_LIBS}
)
