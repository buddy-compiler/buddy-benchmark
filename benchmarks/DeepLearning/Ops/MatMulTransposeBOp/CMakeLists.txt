add_executable(dl-op-linalg-matmul-transpose-b-benchmark
  Main.cpp
)
target_link_libraries(dl-op-linalg-matmul-transpose-b-benchmark GoogleBenchmark)

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT matmul_transpose_b_scalar_O0.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/MatMulTransposeB.mlir |
          sed -e {s/@kernel_placeholder/@matmul_transpose_b_scalar_O0/} |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_transpose_b_scalar_O0.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} matmul_transpose_b_scalar_O0.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_transpose_b_scalar_O0.o
  VERBATIM)

add_library(matmul_transpose_b_scalar_O0 STATIC matmul_transpose_b_scalar_O0.o)
set_target_properties(matmul_transpose_b_scalar_O0 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-transpose-b-benchmark
  matmul_transpose_b_scalar_O0
)

add_custom_command(OUTPUT matmul_transpose_b_scalar_O3.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/MatMulTransposeB.mlir |
          sed -e {s/@kernel_placeholder/@matmul_transpose_b_scalar_O3/} |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_transpose_b_scalar_O3.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_transpose_b_scalar_O3.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_transpose_b_scalar_O3.o
  VERBATIM)

add_library(matmul_transpose_b_scalar_O3 STATIC matmul_transpose_b_scalar_O3.o)
set_target_properties(matmul_transpose_b_scalar_O3 PROPERTIES LINKER_LANGUAGE CXX)
target_link_directories(dl-op-linalg-matmul-transpose-b-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(dl-op-linalg-matmul-transpose-b-benchmark
  matmul_transpose_b_scalar_O3
)

add_custom_command(OUTPUT matmul_transpose_b_scalar_O3_omp.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/MatMulTransposeB.mlir |
          sed -e {s/@kernel_placeholder/@matmul_transpose_b_scalar_O3_omp/} |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_transpose_b_scalar_O3_omp.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_transpose_b_scalar_O3_omp.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_transpose_b_scalar_O3_omp.o
  VERBATIM)

add_library(matmul_transpose_b_scalar_O3_omp STATIC matmul_transpose_b_scalar_O3_omp.o)
set_target_properties(matmul_transpose_b_scalar_O3_omp PROPERTIES LINKER_LANGUAGE CXX)
target_link_directories(dl-op-linalg-matmul-transpose-b-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(dl-op-linalg-matmul-transpose-b-benchmark
  matmul_transpose_b_scalar_O3_omp
  omp
)

add_custom_command(OUTPUT matmul_transpose_b_vec.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/MatMulTransposeB.mlir |
          sed -e {s/@kernel_placeholder/@matmul_transpose_b_vec/} |
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
            -matmul-transpose-b-vectorization
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_transpose_b_vec.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_transpose_b_vec.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_transpose_b_vec.o
  VERBATIM)

add_library(matmul_transpose_b_vec STATIC matmul_transpose_b_vec.o)
set_target_properties(matmul_transpose_b_vec PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-transpose-b-benchmark
  matmul_transpose_b_vec
)

# Build the target for your new method here.
