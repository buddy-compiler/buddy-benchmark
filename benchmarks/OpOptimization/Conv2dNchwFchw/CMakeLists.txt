add_custom_command(OUTPUT mlir-conv2d.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/opt_conv2d.mlir 
            --lower-affine
            -convert-vector-to-llvm
            -convert-memref-to-llvm
            -convert-scf-to-cf
            -convert-linalg-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNchwFchw/mlir-conv2d.o
)
add_library(MLIRCONV2D STATIC mlir-conv2d.o)
set_target_properties(MLIRCONV2D PROPERTIES LINKER_LANGUAGE CXX)

add_executable(conv2d-benchmark
  Main.cpp
  MLIROptBenchmark.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_link_libraries(conv2d-benchmark
  GoogleBenchmark
  MLIRCONV2D 
  ${OpenCV_LIBS}
  )
