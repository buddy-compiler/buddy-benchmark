add_custom_command(OUTPUT mlir-conv2d-ocv.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/Conv2DNchwFchw.mlir |
          sed 's/conv2d_nchw_fchw/conv2d_nchw_fchw_ocv/' |
          ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
            -conv-optimize="kernel-m=2;kernel-n=3;vec-size=16"
            -lower-affine
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-linalg-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNchwFchw/mlir-conv2d-ocv.o
)
add_library(Conv2DNchwFchwOCV STATIC mlir-conv2d-ocv.o)

add_custom_command(OUTPUT mlir-conv2d-broadcast.o
  COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/Conv2DNchwFchwBroadcast.mlir
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
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNchwFchw/mlir-conv2d-broadcast.o
)
add_library(Conv2DNchwFchwBroadcast STATIC mlir-conv2d-broadcast.o)

add_custom_command(OUTPUT mlir-conv2d-scalar.o
  COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/Conv2DNchwFchw.mlir |
          sed 's/conv2d_nchw_fchw/conv2d_nchw_fchw_scalar/' |
          ${LLVM_MLIR_BINARY_DIR}/mlir-opt
            -convert-linalg-to-affine-loops
            -lower-affine
            -convert-vector-to-llvm
            -finalize-memref-to-llvm
            -convert-scf-to-cf
            -convert-linalg-to-llvm
            -llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNchwFchw/mlir-conv2d-scalar.o
)
add_library(Conv2DNchwFchwScalar STATIC mlir-conv2d-scalar.o)

set_target_properties(Conv2DNchwFchwOCV PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(Conv2DNchwFchwBroadcast PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(Conv2DNchwFchwScalar PROPERTIES LINKER_LANGUAGE CXX)

add_executable(conv2d-nchw-fchw-benchmark
  Main.cpp
  Conv2DNchwFchwBenchmark.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_link_libraries(conv2d-nchw-fchw-benchmark
  GoogleBenchmark
  Conv2DNchwFchwOCV
  Conv2DNchwFchwScalar
  Conv2DNchwFchwBroadcast
  )
