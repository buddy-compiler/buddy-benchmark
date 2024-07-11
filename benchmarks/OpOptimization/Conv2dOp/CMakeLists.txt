if (CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

add_custom_command(OUTPUT conv2d_scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dOp/Conv2D.mlir |
  sed 's/@conv_2d/@conv_2d_scalar/' |
  ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        -convert-linalg-to-loops 
        -lower-affine 
        -arith-bufferize 
        -convert-scf-to-cf 
        -convert-vector-to-llvm 
        -convert-arith-to-llvm 
        -finalize-memref-to-llvm
        -llvm-request-c-wrappers 
        -convert-func-to-llvm 
        -reconcile-unrealized-casts | 
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
	            -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dOp/conv2d_scalar.o
)
add_library(Conv2DScalar STATIC conv2d_scalar.o)
set_target_properties(Conv2DScalar PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT conv2d_rvv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dOp/Conv2DRVV.mlir |
        sed 's/@conv_2d/@conv_2d_rvv/' |
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
        ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate --buddy-to-llvmir | 
        ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
                -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dOp/conv2d_rvv.o
)
add_library(Conv2DRVV STATIC conv2d_rvv.o)
set_target_properties(Conv2DRVV PROPERTIES LINKER_LANGUAGE CXX)

add_executable(conv2d-benchmark
  Conv2DBenchmark.cpp
  )

set_target_properties(conv2d-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(conv2d-benchmark
  ${BenchmarkTool}
  Conv2DScalar
  Conv2DRVV
  )
