separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT softmaxexpsumdiv_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/SoftmaxExpSumDivOp/SoftmaxExpSumDiv.mlir |
          sed 's/@softmaxexpsumdiv/@softmaxexpsumdiv_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o softmaxexpsumdiv_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} softmaxexpsumdiv_scalar.ll
            -c -save-temps -o softmaxexpsumdiv_scalar.o
)
add_library(SoftmaxExpSumDivScalar STATIC softmaxexpsumdiv_scalar.o)
set_target_properties(SoftmaxExpSumDivScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT softmaxexpsumdiv_auto_vectorization.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/SoftmaxExpSumDivOp/SoftmaxExpSumDiv.mlir |
          sed 's/@softmaxexpsumdiv/@softmaxexpsumdiv_auto_vectorization/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -lower-affine
            -finalize-memref-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -convert-math-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o softmaxexpsumdiv_auto_vectorization.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} softmaxexpsumdiv_auto_vectorization.ll
            -c -save-temps -o softmaxexpsumdiv_auto_vectorization.o
)
add_library(SoftmaxExpSumDivAutoVectorization STATIC softmaxexpsumdiv_auto_vectorization.o)
set_target_properties(SoftmaxExpSumDivAutoVectorization PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-softmax-exp-sum-div-benchmark
  GoogleBenchmarkMain.cpp
  )

set_target_properties(dl-op-linalg-softmax-exp-sum-div-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-softmax-exp-sum-div-benchmark
  ${BenchmarkTool}
  SoftmaxExpSumDivScalar
  SoftmaxExpSumDivAutoVectorization
)
