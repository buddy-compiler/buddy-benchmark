add_custom_command(
  OUTPUT 
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir 
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMAND
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/buddy_tinyllama_import.py
  COMMENT "Generating forward.mlir, subgraph0.mlir..."
)

add_executable(dl-model-tinyllama-benchmark
  Main.cpp
)

target_link_libraries(dl-model-tinyllama-benchmark
  GoogleBenchmark
)

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

################################################################################
#
# Build scalar target.
#
################################################################################
add_custom_command(
  OUTPUT forward_scalar.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir |
    sed -e {s/@forward/@forward_scalar/} |
    sed -e {s/@subgraph0/@subgraph0_scalar/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -expand-strided-metadata
    -finalize-memref-to-llvm
    -llvm-request-c-wrappers
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward_scalar.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward_scalar.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
  COMMENT "Building forward_scalar.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_scalar.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_scalar/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -pass-pipeline 
    "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -eliminate-empty-tensors
    -empty-tensor-to-alloc-tensor
    -one-shot-bufferize
    -convert-linalg-to-affine-loops
    -lower-affine
    -func-bufferize
    -tensor-bufferize
    -arith-bufferize
    -buffer-deallocation
    -finalizing-bufferize
    -convert-vector-to-scf
    -expand-strided-metadata
    -convert-vector-to-llvm
    -finalize-memref-to-llvm
    -convert-scf-to-cf
    -llvm-request-c-wrappers
    -convert-arith-to-llvm
    -convert-math-to-llvm
    -convert-math-to-libm 
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph0_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph0_scalar.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_scalar.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0_scalar.o"
  VERBATIM)
add_library(tinyllama_scalar STATIC subgraph0_scalar.o forward_scalar.o)
set_target_properties(tinyllama_scalar PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-model-tinyllama-benchmark
  tinyllama_scalar
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)

################################################################################
#
# Build matmul/batch_matmul optimization target.
#
################################################################################
add_custom_command(
  OUTPUT forward_matmul_opt.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir |
    sed -e {s/@forward/@forward_matmul_opt/} |
    sed -e {s/@subgraph0/@subgraph0_matmul_opt/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -expand-strided-metadata
    -finalize-memref-to-llvm
    -llvm-request-c-wrappers
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward_matmul_opt.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward_matmul_opt.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward_matmul_opt.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
  COMMENT "Building forward_matmul_opt.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_matmul_opt.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_matmul_opt/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -pass-pipeline 
    "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
    -eliminate-empty-tensors
    -empty-tensor-to-alloc-tensor
    -one-shot-bufferize
    -matmul-parallel-vectorization-optimize # matmul optimization
    -batchmatmul-optimize # batchmatmul optimization
    -convert-linalg-to-affine-loops
    -lower-affine
    -func-bufferize
    -tensor-bufferize
    -arith-bufferize
    -buffer-deallocation
    -finalizing-bufferize
    -convert-vector-to-scf
    -expand-strided-metadata
    -convert-vector-to-llvm
    -finalize-memref-to-llvm
    -convert-scf-to-cf
    -llvm-request-c-wrappers
    -convert-arith-to-llvm
    -convert-math-to-llvm
    -convert-math-to-libm 
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph0_matmul_opt.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph0_matmul_opt.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_matmul_opt.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0_matmul_opt.o"
  VERBATIM)
add_library(tinyllama_matmul_opt STATIC subgraph0_matmul_opt.o forward_matmul_opt.o)
set_target_properties(tinyllama_matmul_opt PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-model-tinyllama-benchmark
  tinyllama_matmul_opt
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)

################################################################################
#
# Build matmul/batch_matmul optimization target with openmp.
#
################################################################################
add_custom_command(
  OUTPUT forward_matmul_opt_omp.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir |
    sed -e {s/@forward/@forward_matmul_opt_omp/} |
    sed -e {s/@subgraph0/@subgraph0_matmul_opt_omp/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -expand-strided-metadata
    -finalize-memref-to-llvm
    -llvm-request-c-wrappers
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward_matmul_opt_omp.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward_matmul_opt_omp.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward_matmul_opt_omp.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
  COMMENT "Building forward_matmul_opt_omp.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_matmul_opt_omp.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_matmul_opt_omp/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -pass-pipeline 
    "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),func.func(tosa-to-tensor),func.func(tosa-to-arith))" |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
    -eliminate-empty-tensors
    -empty-tensor-to-alloc-tensor
    -one-shot-bufferize
    -matmul-parallel-vectorization-optimize #  # matmul optimization
    -batchmatmul-optimize #  # batchmatmul optimization
    -convert-linalg-to-affine-loops
    -lower-affine
    -convert-scf-to-openmp # openmp support
    -func-bufferize
    -tensor-bufferize
    -arith-bufferize
    -buffer-deallocation
    -finalizing-bufferize
    -convert-vector-to-scf
    -expand-strided-metadata
    -convert-vector-to-llvm
    -finalize-memref-to-llvm
    -convert-scf-to-cf
    -llvm-request-c-wrappers
    -convert-openmp-to-llvm # openmp support
    -convert-arith-to-llvm
    -convert-math-to-llvm
    -convert-math-to-libm 
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph0_matmul_opt_omp.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph0_matmul_opt_omp.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_matmul_opt_omp.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0_matmul_opt_omp.o"
  VERBATIM)
add_library(tinyllama_matmul_opt_omp subgraph0_matmul_opt_omp.o forward_matmul_opt_omp.o)
set_target_properties(tinyllama_matmul_opt_omp PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(dl-model-tinyllama-benchmark
  ${OMP_SHARED_DIR}
  tinyllama_matmul_opt_omp
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)
