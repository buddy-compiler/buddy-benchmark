add_custom_command(OUTPUT mlir-conv2d_nhwc_hwcf_scalar.o
  COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/Conv2DNhwcHwcf.mlir |
          sed 's/conv2d_nhwc_hwcf/conv2d_nhwc_hwcf_scalar/' |
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
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
            --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNhwcHwcf/mlir-conv2d_nhwc_hwcf_scalar.o 
)
add_library(Conv2dNhwcHwcfScalar STATIC mlir-conv2d_nhwc_hwcf_scalar.o)
set_target_properties(Conv2dNhwcHwcfScalar PROPERTIES LINKER_LANGUAGE CXX)

function(build_conv2d_nhwc_hwcf_broadcast step)
  add_custom_command(OUTPUT mlir_conv2d_nhwc_hwfc_${step}.o
    COMMAND 
      cat ${CMAKE_CURRENT_SOURCE_DIR}/Conv2DNhwcHwcf.mlir |
      sed 's/conv2d_nhwc_hwcf/conv2d_nhwc_hwcf_broadcast_${step}/' |
      ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        -conv-broadcast=stride=${step}
        -convert-linalg-to-loops
        -convert-vector-to-scf
        -lower-affine
        -convert-scf-to-cf
        -convert-vector-to-llvm
        -finalize-memref-to-llvm
        -convert-arith-to-llvm
        -llvm-request-c-wrappers
        -convert-func-to-llvm
        -convert-cf-to-llvm
        -reconcile-unrealized-casts |
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
        --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNhwcHwcf/mlir_conv2d_nhwc_hwfc_${step}.o
  )
  add_library(Conv2dNhwcHwcfBroadcast${step} STATIC mlir_conv2d_nhwc_hwfc_${step}.o)
  set_target_properties(Conv2dNhwcHwcfBroadcast${step} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_conv2d_nhwc_hwcf_broadcast(16)
build_conv2d_nhwc_hwcf_broadcast(32)
build_conv2d_nhwc_hwcf_broadcast(64)
build_conv2d_nhwc_hwcf_broadcast(128)
build_conv2d_nhwc_hwcf_broadcast(256)

add_custom_command(OUTPUT mlir_conv2d_nhwc_hwfc_broadcast.o
  COMMAND
  ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt ${CMAKE_CURRENT_SOURCE_DIR}/Conv2DNhwcHwcfBroadcast.mlir
    -convert-linalg-to-loops
    -convert-vector-to-scf
    -lower-affine
    -convert-scf-to-cf
    -convert-vector-to-llvm
    -finalize-memref-to-llvm
    -convert-arith-to-llvm
    -llvm-request-c-wrappers
    -convert-func-to-llvm
    -convert-cf-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
  --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/Conv2dNhwcHwcf/mlir_conv2d_nhwc_hwfc_broadcast.o
)

add_library(Conv2dNhwcHwcfBroadcast STATIC mlir_conv2d_nhwc_hwfc_broadcast.o)
set_target_properties(Conv2dNhwcHwcfBroadcast PROPERTIES LINKER_LANGUAGE CXX)

add_executable(conv2d-nhwc-hwcf-benchmark
  Main.cpp
  Conv2DNhwcHwcfBenchmark.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
target_link_directories(conv2d-nhwc-hwcf-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(conv2d-nhwc-hwcf-benchmark
  GoogleBenchmark
  Conv2dNhwcHwcfScalar
  Conv2dNhwcHwcfBroadcast
  Conv2dNhwcHwcfBroadcast16
  Conv2dNhwcHwcfBroadcast32
  Conv2dNhwcHwcfBroadcast64
  Conv2dNhwcHwcfBroadcast128
  Conv2dNhwcHwcfBroadcast256
  mlir_c_runner_utils
  mlir_runner_utils
)