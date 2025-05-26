add_executable(dl-op-tosa-transpose-benchmark
  Main.cpp
)
target_link_libraries(dl-op-tosa-transpose-benchmark GoogleBenchmark)

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT transpose_scalar_O0.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/Transpose2D.mlir |
          sed -e {s/@transpose_2d_placeholder/@transpose_2d_scalar_O0/} |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -pass-pipeline 
            "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize
              -matmul-parallel-vectorization-optimize
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -affine-parallelize
              -lower-affine
              -convert-scf-to-openmp
              -func-bufferize
              -arith-bufferize
              -tensor-bufferize
              -buffer-deallocation
              -finalizing-bufferize
              -convert-vector-to-scf
              -expand-strided-metadata
              -convert-vector-to-llvm
              -memref-expand
              -arith-expand
              -convert-arith-to-llvm
              -finalize-memref-to-llvm
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              -convert-openmp-to-llvm
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o transpose_scalar_O0.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} transpose_scalar_O0.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/transpose_scalar_O0.o
  VERBATIM)

add_library(transpose_scalar_O0 STATIC transpose_scalar_O0.o)
set_target_properties(transpose_scalar_O0 PROPERTIES LINKER_LANGUAGE CXX)
target_link_directories(dl-op-tosa-transpose-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(dl-op-tosa-transpose-benchmark
  transpose_scalar_O0
  omp
)

add_custom_command(OUTPUT transpose_scalar_O3.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/Transpose2D.mlir |
          sed -e {s/@transpose_2d_placeholder/@transpose_2d_scalar_O3/} |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -pass-pipeline 
            "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize
              -matmul-parallel-vectorization-optimize
              -batchmatmul-optimize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -affine-parallelize
              -lower-affine
              -convert-scf-to-openmp
              -func-bufferize
              -arith-bufferize
              -tensor-bufferize
              -buffer-deallocation
              -finalizing-bufferize
              -convert-vector-to-scf
              -expand-strided-metadata
              -convert-vector-to-llvm
              -memref-expand
              -arith-expand
              -convert-arith-to-llvm
              -finalize-memref-to-llvm
              -convert-scf-to-cf
              -llvm-request-c-wrappers
              -convert-openmp-to-llvm
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o transpose_scalar_O3.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} transpose_scalar_O3.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/transpose_scalar_O3.o
  VERBATIM)

add_library(transpose_scalar_O3 STATIC transpose_scalar_O3.o)
set_target_properties(transpose_scalar_O3 PROPERTIES LINKER_LANGUAGE CXX)
target_link_directories(dl-op-tosa-transpose-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(dl-op-tosa-transpose-benchmark
  transpose_scalar_O3
  omp
)

# Build the target for your new method here.
