#-------------------------------------------------------------------------------
# MLIR Linalg Dialect MatMal Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-matmul.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRMatMul.mlir 
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -convert-linalg-to-llvm
            -llvm-request-c-wrappers
            -convert-memref-to-llvm
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
            -convert-linalg-to-llvm
            -expand-strided-metadata
            -lower-affine
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-memref-to-llvm
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
# MLIR SCF Dialect GccLoopsEx1 Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex1.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx1.mlir 
            -convert-scf-to-cf
            -expand-strided-metadata
            -lower-affine
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-memref-to-llvm
            -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex1.o
)
add_library(MLIRGccLoopsEx1 STATIC mlir-gccloopsex1.o)
set_target_properties(MLIRGccLoopsEx1 PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect MatVec Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex2a.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx2a.mlir 
            -convert-scf-to-cf
            -expand-strided-metadata
            -lower-affine
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-memref-to-llvm
            -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex2a.o
)
add_library(MLIRGccLoopsEx2a STATIC mlir-gccloopsex2a.o)
set_target_properties(MLIRGccLoopsEx2a PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Image Processing Benchmark Target
#-------------------------------------------------------------------------------

add_executable(vectorization-benchmark
  Main.cpp
  MLIRMatMulBenchmark.cpp
  MLIRMatVecBenchmark.cpp
  MLIRGccLoopsEx1Benchmark.cpp
  MLIRGccLoopsEx2aBenchmark.cpp
  )

target_link_libraries(vectorization-benchmark
  GoogleBenchmark
  MLIRMatMul
  MLIRMatVec
  MLIRGccLoopsEx1
  MLIRGccLoopsEx2a
  )
