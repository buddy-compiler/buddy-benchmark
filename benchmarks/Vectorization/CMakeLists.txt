
add_subdirectory(gccloops)
add_subdirectory(linpackc)
add_subdirectory(polybench)

#-------------------------------------------------------------------------------
# MLIR Linalg Dialect MatMal Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-matmul.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRMatMul.mlir 
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-matmul.o
)
add_library(MLIRMatMul STATIC mlir-matmul.o)
set_target_properties(MLIRMatMul PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR Linalg Dialect MatVec Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-matvec.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRMatVec.mlir 
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -expand-strided-metadata
            -lower-affine
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-matvec.o
)
add_library(MLIRMatVec STATIC mlir-matvec.o)
set_target_properties(MLIRMatVec PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Vectorization Benchmark Targets
#-------------------------------------------------------------------------------

add_executable(vectorization-matrix-benchmark
  Main.cpp
  MLIRMatMulBenchmark.cpp
  MLIRMatVecBenchmark.cpp
  )

target_link_libraries(vectorization-matrix-benchmark
  GoogleBenchmark
  MLIRMatMul
  MLIRMatVec
  )
