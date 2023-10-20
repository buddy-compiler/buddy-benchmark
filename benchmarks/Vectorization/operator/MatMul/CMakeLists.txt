#-------------------------------------------------------------------------------
# MLIR Linalg Dialect MatMal Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-matmul.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/operator/MatMul/MLIRMatMul.mlir
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -convert-linalg-to-llvm
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/operator/MatMul/mlir-matmul.o
)
add_library(MLIRMatMul STATIC mlir-matmul.o)
set_target_properties(MLIRMatMul PROPERTIES LINKER_LANGUAGE CXX)



