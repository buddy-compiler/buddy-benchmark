if (CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O0 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/SoftmaxExpSumDivOp/softmaxexpsumdiv_scalar.o
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/SoftmaxExpSumDivOp/softmaxexpsumdiv_auto_vectorization.o
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
