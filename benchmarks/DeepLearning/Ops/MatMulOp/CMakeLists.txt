add_executable(dl-op-linalg-matmul-benchmark
  Main.cpp
)

set(BenchmarkTool GoogleBenchmark)
target_link_libraries(dl-op-linalg-matmul-benchmark
  ${BenchmarkTool}
)

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT matmul_scalar_O0.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/matmul_scalar.mlir |
          sed 's/matmul_scalar/matmul_scalar_O0/' |
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
            COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} matmul_scalar_O0.ll -c -save-temps
                    -o ${CMAKE_CURRENT_BINARY_DIR}/matmul_scalar_O0.o
)
add_library(matmul_scalar_O0 STATIC matmul_scalar_O0.o)
set_target_properties(matmul_scalar_O0 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-op-linalg-matmul-benchmark
  matmul_scalar_O0
)

macro(add_matmul_target TARGET_NAME)
  set(MLIR_FILE "${TARGET_NAME}.mlir")
  set(LL_FILE "${TARGET_NAME}.ll")
  set(O_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.o")
  add_custom_command(OUTPUT ${O_FILE}
    COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulOp/${MLIR_FILE} |
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
            ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o ${LL_FILE}
    COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} ${LL_FILE} -c -save-temps
            -o ${O_FILE}
  )
  add_library(${TARGET_NAME} STATIC ${O_FILE})
  set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(dl-op-linalg-matmul-benchmark ${TARGET_NAME})
endmacro()

add_matmul_target(matmul_scalar)
add_matmul_target(matmul_fixed_vector_scalar)
add_matmul_target(matmul_fixed_vector_mask)
