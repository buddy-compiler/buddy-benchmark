add_custom_command(OUTPUT mlir-conv2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/Conv2DNchwFchw.mlir
            --conv-optimize="kernel-m=2;kernel-n=3;vec-size=16"
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

add_custom_command(OUTPUT mlir-conv2d-org.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/Conv2DNchwFchw.mlir |
          sed 's/conv2d/conv2d_org/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            --convert-linalg-to-affine-loops
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
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNchwFchw/mlir-conv2d-org.o
)
add_library(MLIRCONV2DORG STATIC mlir-conv2d-org.o)

set_target_properties(MLIRCONV2D PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(MLIRCONV2DORG PROPERTIES LINKER_LANGUAGE CXX)

add_executable(conv2d-benchmark
  Main.cpp
  MLIROptBenchmark.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_link_libraries(conv2d-benchmark
  GoogleBenchmark
  MLIRCONV2D 
  MLIRCONV2DORG
  ${OpenCV_LIBS}
  )
