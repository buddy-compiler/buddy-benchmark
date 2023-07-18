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

add_custom_command(OUTPUT mlir-conv2d-im2col.o
  COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/Conv2DNchwFchwIm2col.mlir
            --convert-linalg-to-loops
            --convert-vector-to-scf
            --convert-scf-to-cf
            --convert-linalg-to-llvm
            --expand-strided-metadata
            --lower-affine
            --convert-vector-to-llvm
            --memref-expand
            --arith-expand
            --convert-arith-to-llvm
            --finalize-memref-to-llvm
            --convert-math-to-llvm
            --llvm-request-c-wrappers
            --convert-func-to-llvm
            --reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNchwFchw/mlir-conv2d-im2col.o
)
add_library(Conv2DNchwFchwIm2col STATIC mlir-conv2d-im2col.o)

add_custom_command(OUTPUT mlir-conv2d-winagrad.o
  COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/Conv2DNchwFchwWinagrad.mlir
            --convert-linalg-to-loops
            --expand-strided-metadata
            --lower-affine
            --convert-vector-to-scf
            --convert-scf-to-cf
            --convert-vector-to-llvm
            --memref-expand
            --arith-expand
            --convert-arith-to-llvm
            --convert-index-to-llvm
            --finalize-memref-to-llvm
            --convert-math-to-llvm
            --llvm-request-c-wrappers
            --convert-func-to-llvm
            --reconcile-unrealized-casts |
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj
	    -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNchwFchw/mlir-conv2d-winagrad.o
)
add_library(Conv2DNchwFchwWinagrad STATIC mlir-conv2d-winagrad.o)

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
set_target_properties(Conv2DNchwFchwIm2col PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(Conv2DNchwFchwWinagrad PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(Conv2DNchwFchwScalar PROPERTIES LINKER_LANGUAGE CXX)

function(build_conv2d_broadcast step)
  add_custom_command(OUTPUT mlir-conv2d-broadcast-${step}.o
    COMMAND cat ${BUDDY_SOURCE_DIR}/benchmarks/OpOptimization/Conv2dNchwFchw/Conv2DNchwFchwBroadcast.mlir |
            sed 's/STEP_PLACEHOLDER/${step}/g' |
            ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
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
        -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNchwFchw/mlir-conv2d-broadcast-${step}.o
  )
  add_library(Conv2DNchwFchwBroadcast${step} STATIC mlir-conv2d-broadcast-${step}.o)
  set_target_properties(Conv2DNchwFchwBroadcast${step} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_conv2d_broadcast(16)
build_conv2d_broadcast(32)
build_conv2d_broadcast(64)
build_conv2d_broadcast(128)
build_conv2d_broadcast(256)

add_executable(conv2d-nchw-fchw-benchmark
  Main.cpp
  Conv2DNchwFchwBenchmark.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

target_link_libraries(conv2d-nchw-fchw-benchmark
  GoogleBenchmark
  Conv2DNchwFchwOCV
  Conv2DNchwFchwScalar
  Conv2DNchwFchwIm2col
  Conv2DNchwFchwWinagrad
  Conv2DNchwFchwBroadcast16
  Conv2DNchwFchwBroadcast32
  Conv2DNchwFchwBroadcast64
  Conv2DNchwFchwBroadcast128
  Conv2DNchwFchwBroadcast256
  )
