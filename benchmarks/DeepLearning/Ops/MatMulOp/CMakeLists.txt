add_executable(dl-op-linalg-matmul-benchmark
  Main.cpp
)
target_link_libraries(dl-op-linalg-matmul-benchmark GoogleBenchmark)

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT matmul_scalar_O0.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/matmul.mlir |
          sed 's/@matmul/@matmul_scalar_O0/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_scalar_O0.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} matmul_scalar_O0.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_scalar_O0.o
)
add_library(matmul_scalar_O0 STATIC matmul_scalar_O0.o)
set_target_properties(matmul_scalar_O0 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-benchmark
  matmul_scalar_O0
)

add_custom_command(OUTPUT matmul_scalar_O3.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/matmul.mlir |
          sed 's/@matmul/@matmul_scalar_O3/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_scalar_O3.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_scalar_O3.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_scalar_O3.o
)
add_library(matmul_scalar_O3 STATIC matmul_scalar_O3.o)
set_target_properties(matmul_scalar_O3 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-benchmark
  matmul_scalar_O3
)

add_custom_command(OUTPUT matmul_tile.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/matmul.mlir |
          sed 's/@matmul/@matmul_tile/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -matmul-optimize="vec-size=16;kernel-m=2;kernel-n=4"
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-cf
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-arith-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_tile.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_tile.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_tile.o
)
add_library(matmul_tile STATIC matmul_tile.o)
set_target_properties(matmul_tile PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-benchmark
  matmul_tile
)

add_custom_command(OUTPUT matmul_vec.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/matmul.mlir |
          sed 's/@matmul/@matmul_vec/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -matmul-parallel-vectorization-optimize
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_vec.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_vec.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_vec.o
)
add_library(matmul_vec STATIC matmul_vec.o)
set_target_properties(matmul_vec PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-benchmark
  matmul_vec
)

add_custom_command(OUTPUT matmul_vec_omp.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/matmul.mlir |
          sed 's/@matmul/@matmul_vec_omp/' |
          ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
            -matmul-parallel-vectorization-optimize
            -convert-linalg-to-loops
            -expand-strided-metadata
            -lower-affine
            -convert-scf-to-openmp
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -convert-openmp-to-llvm
            -convert-vector-to-llvm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o matmul_vec_omp.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} matmul_vec_omp.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_vec_omp.o
)
add_library(matmul_vec_omp STATIC matmul_vec_omp.o)
set_target_properties(matmul_vec_omp PROPERTIES LINKER_LANGUAGE CXX)
target_link_directories(dl-op-linalg-matmul-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(dl-op-linalg-matmul-benchmark
  matmul_vec_omp
  ${OMP_SHARED_DIR}
)

# Build the target for your new method here.
