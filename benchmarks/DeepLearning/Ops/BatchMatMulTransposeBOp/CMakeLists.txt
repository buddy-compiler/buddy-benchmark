add_executable(dl-op-batch-matmul-transpose-b-benchmark
  Main.cpp
)
target_link_libraries(dl-op-batch-matmul-transpose-b-benchmark GoogleBenchmark)

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT batch_matmul_transpose_b_scalar_O0.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/BatchMatMulTransposeB.mlir |
          sed -e {s/@kernel_placeholder/@batch_matmul_transpose_b_scalar_O0/} |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -pass-pipeline 
            "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -affine-parallelize
              -lower-affine
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
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o batch_matmul_transpose_b_scalar_O0.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} batch_matmul_transpose_b_scalar_O0.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_transpose_b_scalar_O0.o
  VERBATIM)

add_library(batch_matmul_transpose_b_scalar_O0 STATIC batch_matmul_transpose_b_scalar_O0.o)
set_target_properties(batch_matmul_transpose_b_scalar_O0 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-batch-matmul-transpose-b-benchmark
  batch_matmul_transpose_b_scalar_O0
)

add_custom_command(OUTPUT batch_matmul_transpose_b_scalar_O3.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/BatchMatMulTransposeB.mlir |
          sed -e {s/@kernel_placeholder/@batch_matmul_transpose_b_scalar_O3/} |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -pass-pipeline 
            "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
              -arith-expand
              -eliminate-empty-tensors
              -empty-tensor-to-alloc-tensor
              -one-shot-bufferize
              -convert-linalg-to-affine-loops
              -affine-loop-fusion
              -affine-parallelize
              -lower-affine
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
              -convert-arith-to-llvm
              -convert-math-to-llvm
              -convert-math-to-libm
              -convert-func-to-llvm
              -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o batch_matmul_transpose_b_scalar_O3.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_transpose_b_scalar_O3.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_transpose_b_scalar_O3.o
  VERBATIM)

add_library(batch_matmul_transpose_b_scalar_O3 STATIC batch_matmul_transpose_b_scalar_O3.o)
set_target_properties(batch_matmul_transpose_b_scalar_O3 PROPERTIES LINKER_LANGUAGE CXX)
target_link_directories(dl-op-batch-matmul-transpose-b-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(dl-op-batch-matmul-transpose-b-benchmark
  batch_matmul_transpose_b_scalar_O3
)

add_custom_command(OUTPUT batch_matmul_transpose_b_vec.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/BatchMatMulTransposeB.mlir |
          sed -e {s/@kernel_placeholder/@batch_matmul_transpose_b_vec/} |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -pass-pipeline 
            "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -arith-expand
            -eliminate-empty-tensors
            -empty-tensor-to-alloc-tensor
            -one-shot-bufferize
            -func-bufferize
            -arith-bufferize
            -tensor-bufferize
            -buffer-deallocation
            -finalizing-bufferize
            -batchmatmul-transpose-b-vectorization
            -convert-linalg-to-affine-loops
            -affine-loop-fusion
            -lower-affine
            -convert-vector-to-scf
            -expand-strided-metadata
            -convert-vector-to-llvm
            -memref-expand
            -arith-expand
            -convert-arith-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-arith-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm 
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o batch_matmul_transpose_b_vec.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_transpose_b_vec.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_transpose_b_vec.o
  VERBATIM)

add_library(batch_matmul_transpose_b_vec STATIC batch_matmul_transpose_b_vec.o)
set_target_properties(batch_matmul_transpose_b_vec PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-batch-matmul-transpose-b-benchmark
  batch_matmul_transpose_b_vec
)

# Build the target for your new method here.
