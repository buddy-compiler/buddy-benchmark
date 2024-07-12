add_custom_command(
  OUTPUT ${BUDDY_DEEP_LEARNING_DIR}/Models/MobileNet-V3/arg0.data
  ${BUDDY_DEEP_LEARNING_DIR}/Models/MobileNet-V3/arg1.data
  ${BUDDY_DEEP_LEARNING_DIR}/Models/MobileNet-V3/forward.mlir
  ${BUDDY_DEEP_LEARNING_DIR}/Models/MobileNet-V3/subgraph0.mlir
  COMMAND python3 ${BUDDY_DEEP_LEARNING_DIR}/Models/MobileNet-V3/buddy-mobilenetv3-import.py
  COMMENT "Generating forward.mlir, subgraph0.mlir and parameter files"
)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Models/MobileNet-V3/forward.mlir
  -pass-pipeline
  "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
            empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
            func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
  -pass-pipeline
  "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
            eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
            convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
            convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
            convert-func-to-llvm, reconcile-unrealized-casts)" |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llvm-as |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} -filetype=obj -relocation-model=pic
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Models/MobileNet-V3/forward.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Models/MobileNet-V3/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_MLIR_BINARY_DIR}/buddy-opt ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Models/MobileNet-V3/subgraph0.mlir
  -pass-pipeline
  "builtin.module(func.func(tosa-to-linalg-named, tosa-to-arith, tosa-to-linalg, tosa-to-tensor))" |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  -convert-elementwise-to-linalg
  -func-bufferize-dynamic-offset
  -arith-bufferize
  -func-bufferize
  -tensor-bufferize
  -linalg-bufferize
  -finalizing-bufferize
  -convert-linalg-to-loops
  -lower-affine
  -convert-scf-to-cf
  -llvm-request-c-wrappers
  -convert-math-to-llvm
  -convert-math-to-libm
  -convert-arith-to-llvm
  -convert-func-to-llvm
  -expand-strided-metadata
  -finalize-memref-to-llvm
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llvm-as |
  ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} -filetype=obj -relocation-model=pic
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Models/MobileNet-V3/subgraph0.o
  DEPENDS ${BUDDY_SOURCE_DIR}/benchmarks/DeepLearning/Models/MobileNet-V3/subgraph0.mlir
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(MOBILENETV3 STATIC subgraph0.o forward.o)
set_target_properties(MOBILENETV3 PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-model-mobileNetV3-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-model-mobileNetV3-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

target_link_libraries(dl-model-mobileNetV3-benchmark
  ${BenchmarkTool}
  MOBILENETV3
  ${BUDDY_MLIR_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)
