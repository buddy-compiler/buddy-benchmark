if (CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

add_custom_command(OUTPUT matmul-scalar-rvv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulRVVOp/MatMul.mlir |
          sed 's/@matmul/@matmul_scalar/' |
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/MatMulRVVOp/matmul-scalar-rvv.o
)
add_library(MatMulScalarRVV STATIC matmul-scalar-rvv.o)
set_target_properties(MatMulScalarRVV PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT matmul-rvv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Ops/MatMulRVVOp/MatMulRVV.mlir |
          sed 's/@matmul/@matmul_rvv/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          -llvm-request-c-wrappers
          -lower-affine 
          -convert-scf-to-cf 
          -convert-math-to-llvm 
          -lower-vector-exp 
          -lower-rvv 
          -convert-vector-to-llvm 
          -finalize-memref-to-llvm 
          -convert-func-to-llvm 
          -reconcile-unrealized-casts | 
            ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	          -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Ops/MatMulRVVOp/matmul-rvv.o
)
add_library(MatMulRVV STATIC matmul-rvv.o)
set_target_properties(MatMulRVV PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-linalg-matmul-benchmark-rvv
  GoogleBenchmarkMain.cpp
  )

set_target_properties(dl-op-linalg-matmul-benchmark-rvv PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-op-linalg-matmul-benchmark-rvv
  ${BenchmarkTool}
  MatMulScalarRVV
  MatMulRVV
  )
