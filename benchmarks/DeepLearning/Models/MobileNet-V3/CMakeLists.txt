add_custom_command(
  OUTPUT 
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir 
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMAND
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/buddy_mobilenetv3_import.py
  COMMENT "Generating forward.mlir, subgraph0.mlir..."
)

add_executable(dl-model-mobilenetv3-benchmark
  GoogleBenchmarkMain.cpp
)
set_target_properties(dl-model-mobilenetv3-benchmark PROPERTIES
  LINK_FLAGS "-static"
)
target_link_libraries(dl-model-mobilenetv3-benchmark
  GoogleBenchmark
)

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

################################################################################
#
# Build scalar target.
#
################################################################################
add_custom_command(
  OUTPUT forward_scalar.o
  COMMAND cat ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/forward.mlir |
    sed -e {s/@forward/@forward_scalar/} -e {s/@subgraph0/@subgraph0_scalar/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -expand-strided-metadata
    -finalize-memref-to-llvm
    -llvm-request-c-wrappers
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward_scalar.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward_scalar.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
  COMMENT "Building forward_scalar.o"
  VERBATIM)
add_custom_command(
  OUTPUT subgraph0_scalar.o
  COMMAND cat ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_scalar/} |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt 
  -pass-pipeline
  "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
    -convert-elementwise-to-linalg
    -linalg-named-op-conversion
    -convert-math-to-llvm
    -convert-math-to-libm
    -one-shot-bufferize
    -convert-linalg-to-affine-loops
    -lower-affine
    -func-bufferize
    -tensor-bufferize
    -arith-bufferize
    -finalizing-bufferize
    -convert-vector-to-scf
    -expand-strided-metadata
    -convert-vector-to-llvm
    -finalize-memref-to-llvm
    -convert-scf-to-cf
    -llvm-request-c-wrappers
    -convert-arith-to-llvm
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph0_scalar.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph0_scalar.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_scalar.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0_scalar.o"
  VERBATIM)
add_library(mobilenetv3_scalar STATIC subgraph0_scalar.o forward_scalar.o)
set_target_properties(mobilenetv3_scalar PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-model-mobilenetv3-benchmark
  mobilenetv3_scalar
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)

################################################################################
#
# Build conv-nhwc-fhwc/depthwise-conv-nhwc-hwc optimization optimization target.
#
################################################################################
add_custom_command(
  OUTPUT forward_conv_opt.o
  COMMAND cat ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/forward.mlir |
    sed -e {s/@forward/@forward_conv_opt/} 
        -e {s/@subgraph0/@subgraph0_conv_opt/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
    -expand-strided-metadata
    -finalize-memref-to-llvm
    -llvm-request-c-wrappers
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o forward_conv_opt.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} forward_conv_opt.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/forward_conv_opt.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
  COMMENT "Building forward_conv_opt.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_conv_opt.o
  COMMAND cat ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_conv_opt/} |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
    -pass-pipeline
    "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
    -convert-elementwise-to-linalg
    -linalg-named-op-conversion
    -convert-math-to-llvm
    -convert-math-to-libm
    -one-shot-bufferize
    -conv-nhwc-fhwc-optimize # conv-nhwc-fhwc optimization
    -depthwise-conv-nhwc-hwc-optimize # depthwise-conv-nhwc-hwc optimization
    -convert-linalg-to-affine-loops
    -lower-affine
    -func-bufferize
    -tensor-bufferize
    -arith-bufferize
    -finalizing-bufferize
    -convert-vector-to-scf
    -expand-strided-metadata
    -convert-vector-to-llvm
    -finalize-memref-to-llvm
    -convert-scf-to-cf
    -llvm-request-c-wrappers
    -convert-arith-to-llvm
    -convert-func-to-llvm
    -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir -o subgraph0_conv_opt.ll
  COMMAND ${LLVM_MLIR_BINARY_DIR}/clang -O3 ${CLANG_FLAGS_LIST} subgraph0_conv_opt.ll
    -c -save-temps -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_conv_opt.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0_conv_opt.o"
  VERBATIM)
add_library(mobilenetv3_conv_opt STATIC subgraph0_conv_opt.o forward_conv_opt.o)
set_target_properties(mobilenetv3_conv_opt PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(dl-model-mobilenetv3-benchmark
  mobilenetv3_conv_opt
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)
