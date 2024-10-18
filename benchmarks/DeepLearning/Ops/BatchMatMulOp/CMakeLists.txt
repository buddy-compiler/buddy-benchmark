add_executable(dl-op-linalg-batch-matmul-benchmark
  Main.cpp
)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark GoogleBenchmark)

# CMAKE_C_FLAGS is set when configuring CMake
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

# Batch MatMul Scalar
add_custom_command(OUTPUT batch_matmul_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMul.mlir |
          sed 's/@batch_matmul/@batch_matmul_scalar/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -convert-linalg-to-loops
          -expand-strided-metadata
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts|
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o batch_matmul_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} batch_matmul_scalar.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_scalar.o
)
add_library(batch_matmul_scalar STATIC batch_matmul_scalar.o)
set_target_properties(batch_matmul_scalar PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark
  batch_matmul_scalar
)

# Batch MatMul Auto Vectorization
add_custom_command(OUTPUT batch_matmul_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMul.mlir |
          sed 's/@batch_matmul/@batch_matmul_auto_vectorization/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -convert-linalg-to-loops
          -expand-strided-metadata
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts|
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o batch_matmul_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_auto_vectorization.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_auto_vectorization.o
)
add_library(batch_matmul_auto_vectorization STATIC batch_matmul_auto_vectorization.o)
set_target_properties(batch_matmul_auto_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark
  batch_matmul_auto_vectorization
)

# BatchMatMul Vectorization
add_custom_command(OUTPUT batch_matmul_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMulVec.mlir |
          sed 's/@batch_matmul/@batch_matmul_vectorization/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -convert-linalg-to-loops
          -expand-strided-metadata
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts|
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o batch_matmul_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_vectorization.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_vectorization.o
)
add_library(batch_matmul_vectorization STATIC batch_matmul_vectorization.o)
set_target_properties(batch_matmul_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark
  batch_matmul_vectorization
)

# BatchMatMul Tiling Vectorization
add_custom_command(OUTPUT batch_matmul_tilling_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMulVecTile.mlir |
          sed 's/@batch_matmul/@batch_matmul_tile/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -convert-linalg-to-loops
          -expand-strided-metadata
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts|
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o batch_matmul_tilling_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_tilling_vectorization.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_tilling_vectorization.o
)
add_library(batch_matmul_tilling_vectorization STATIC batch_matmul_tilling_vectorization.o)
set_target_properties(batch_matmul_tilling_vectorization PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark
  batch_matmul_tilling_vectorization
)

# BatchMatMul SCF
add_custom_command(OUTPUT batch_matmul_scf.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/BatchMatMulOp/BatchMatMulSCF.mlir |
          sed 's/@batch_matmul/@batch_matmul_scf/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -convert-linalg-to-loops
          -expand-strided-metadata
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -llvm-request-c-wrappers
          -convert-func-to-llvm 
          -reconcile-unrealized-casts|
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir -o batch_matmul_scf.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} batch_matmul_scf.ll
          -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/batch_matmul_scf.o
)
add_library(batch_matmul_scf STATIC batch_matmul_scf.o)
set_target_properties(batch_matmul_scf PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-batch-matmul-benchmark
  batch_matmul_scf
)
