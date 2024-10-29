add_executable(dl-op-linalg-batch-matmul-benchmark-rvv
  Main.cpp
)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark-rvv GoogleBenchmark)

# CMAKE_C_FLAGS is set when configuring CMake
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT batch_matmul_scalar_rvv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/BatchMatMul.mlir |
          sed 's/@batch_matmul/@batch_matmul_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts |
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o batch_matmul_scalar_rvv.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} batch_matmul_scalar_rvv.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_scalar_rvv.o
)
add_library(batch_matmul_scalar_rvv STATIC batch_matmul_scalar_rvv.o)
set_target_properties(batch_matmul_scalar_rvv PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark-rvv
  batch_matmul_scalar_rvv
)

add_custom_command(OUTPUT batch_matmul_auto_vectorization_rvv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/BatchMatMul.mlir |
          sed 's/@batch_matmul/@batch_matmul_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o batch_matmul_auto_vectorization_rvv.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_auto_vectorization_rvv.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_auto_vectorization_rvv.o
)
add_library(batch_matmul_auto_vectorization_rvv STATIC batch_matmul_auto_vectorization_rvv.o)
set_target_properties(batch_matmul_auto_vectorization_rvv PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark-rvv
batch_matmul_auto_vectorization_rvv
)

add_custom_command(OUTPUT batch_matmul_rvv_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/BatchMatMulVectorizationRVV.mlir |
          sed 's/@batch_matmul/@batch_matmul_rvv_vectorization/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -lower-vector-exp 
          -lower-rvv 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o batch_matmul_rvv_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_rvv_vectorization.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_rvv_vectorization.o
)
add_library(batch_matmul_rvv_vectorization STATIC batch_matmul_rvv_vectorization.o)
set_target_properties(batch_matmul_rvv_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark-rvv
  batch_matmul_rvv_vectorization
)

add_custom_command(OUTPUT batch_matmul_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulInt32Op/BatchMatMulVectorization.mlir |
          sed 's/@batch_matmul/@batch_matmul_vectorization/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o batch_matmul_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_vectorization.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_vectorization.o
)
add_library(batch_matmul_vectorization_rvv STATIC batch_matmul_vectorization.o)
set_target_properties(batch_matmul_vectorization_rvv PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark-rvv
batch_matmul_vectorization_rvv
)
