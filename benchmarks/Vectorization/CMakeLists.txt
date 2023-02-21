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
# MLIR SCF Dialect GccLoopsEx2a Operation + Upstream Lowering Passes
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
# MLIR SCF Dialect GccLoopsEx2b Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex2b.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx2b.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex2b.o
)
add_library(MLIRGccLoopsEx2b STATIC mlir-gccloopsex2b.o)
set_target_properties(MLIRGccLoopsEx2b PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx3 Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex3.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx3.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex3.o
)
add_library(MLIRGccLoopsEx3 STATIC mlir-gccloopsex3.o)
set_target_properties(MLIRGccLoopsEx3 PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx4a Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex4a.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx4a.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex4a.o
)
add_library(MLIRGccLoopsEx4a STATIC mlir-gccloopsex4a.o)
set_target_properties(MLIRGccLoopsEx4a PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx4b Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex4b.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx4b.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex4b.o
)
add_library(MLIRGccLoopsEx4b STATIC mlir-gccloopsex4b.o)
set_target_properties(MLIRGccLoopsEx4b PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx4c Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex4c.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx4c.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex4c.o
)
add_library(MLIRGccLoopsEx4c STATIC mlir-gccloopsex4c.o)
set_target_properties(MLIRGccLoopsEx4c PROPERTIES LINKER_LANGUAGE CXX)


#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx7 Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex7.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx7.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex7.o
)
add_library(MLIRGccLoopsEx7 STATIC mlir-gccloopsex7.o)
set_target_properties(MLIRGccLoopsEx7 PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx8 Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex8.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx8.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex8.o
)
add_library(MLIRGccLoopsEx8 STATIC mlir-gccloopsex8.o)
set_target_properties(MLIRGccLoopsEx8 PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx9 Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex9.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx9.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex9.o
)
add_library(MLIRGccLoopsEx9 STATIC mlir-gccloopsex9.o)
set_target_properties(MLIRGccLoopsEx9 PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx10a Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex10a.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx10a.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex10a.o
)
add_library(MLIRGccLoopsEx10a STATIC mlir-gccloopsex10a.o)
set_target_properties(MLIRGccLoopsEx10a PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx10b Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex10b.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx10b.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex10b.o
)
add_library(MLIRGccLoopsEx10b STATIC mlir-gccloopsex10b.o)
set_target_properties(MLIRGccLoopsEx10b PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx11 Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex11.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx11.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex11.o
)
add_library(MLIRGccLoopsEx11 STATIC mlir-gccloopsex11.o)
set_target_properties(MLIRGccLoopsEx11 PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR SCF Dialect GccLoopsEx12 Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-gccloopsex12.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/MLIRGccLoopsEx12.mlir 
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
            -o ${BUDDY_BINARY_DIR}/../benchmarks/Vectorization/mlir-gccloopsex12.o
)
add_library(MLIRGccLoopsEx12 STATIC mlir-gccloopsex12.o)
set_target_properties(MLIRGccLoopsEx12 PROPERTIES LINKER_LANGUAGE CXX)


#-------------------------------------------------------------------------------
# Image Processing Benchmark Target
#-------------------------------------------------------------------------------

add_executable(vectorization-benchmark
  Main.cpp
  MLIRMatMulBenchmark.cpp
  MLIRMatVecBenchmark.cpp
  MLIRGccLoopsEx1Benchmark.cpp
  MLIRGccLoopsEx2aBenchmark.cpp
  MLIRGccLoopsEx2bBenchmark.cpp
  MLIRGccLoopsEx3Benchmark.cpp
  MLIRGccLoopsEx4aBenchmark.cpp
  MLIRGccLoopsEx4bBenchmark.cpp
  MLIRGccLoopsEx4cBenchmark.cpp
  MLIRGccLoopsEx7Benchmark.cpp
  MLIRGccLoopsEx8Benchmark.cpp
  MLIRGccLoopsEx9Benchmark.cpp
  MLIRGccLoopsEx10aBenchmark.cpp
  MLIRGccLoopsEx10bBenchmark.cpp
  MLIRGccLoopsEx11Benchmark.cpp
  MLIRGccLoopsEx12Benchmark.cpp
  )

target_link_libraries(vectorization-benchmark
  GoogleBenchmark
  MLIRMatMul
  MLIRMatVec
  MLIRGccLoopsEx1
  MLIRGccLoopsEx2a
  MLIRGccLoopsEx2b
  MLIRGccLoopsEx3
  MLIRGccLoopsEx4a
  MLIRGccLoopsEx4b
  MLIRGccLoopsEx4c
  MLIRGccLoopsEx7
  MLIRGccLoopsEx8
  MLIRGccLoopsEx9
  MLIRGccLoopsEx10a
  MLIRGccLoopsEx10b
  MLIRGccLoopsEx11
  MLIRGccLoopsEx12
  )
