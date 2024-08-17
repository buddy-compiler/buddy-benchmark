add_custom_command(
  OUTPUT
  ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/forward.mlir
  ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/subgraph0.mlir
  COMMAND python3 ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/buddy_mobilenetv3_import.py
  COMMENT "Generating forward.mlir, subgraph0.mlir"
)

add_custom_command(
  OUTPUT forward_autoVectorization.o
  COMMAND cat ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/forward.mlir |
  sed -e {s/@forward/@forward_autoVectorization/} -e {s/@subgraph0/@subgraph0_autoVectorization/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
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
  -mattr=${BUDDY_OPT_ATTR} -filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Models/MobileNet-V3/forward_autoVectorization.o
  DEPENDS ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/forward.mlir
  COMMENT "Building forward_autoVectorization.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_autoVectorization.o
  COMMAND cat ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/subgraph0.mlir |
  sed -e {s/@subgraph0/@subgraph0_autoVectorization/} |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt 
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
  -mattr=${BUDDY_OPT_ATTR} -filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Models/MobileNet-V3/subgraph0_autoVectorization.o
  DEPENDS ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/subgraph0.mlir
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  COMMENT "Building subgraph0_autoVectorization.o"
  VERBATIM)

add_custom_command(
  OUTPUT forward_vectorization.o
  COMMAND cat ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/forward.mlir |
  sed -e {s/@forward/@forward_vectorization/} -e {s/@subgraph0/@subgraph0_vectorization/} |
  ${LLVM_MLIR_BINARY_DIR}/mlir-opt
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
  -mattr=${BUDDY_OPT_ATTR} -filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Models/MobileNet-V3/forward_vectorization.o
  DEPENDS ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/forward.mlir
  COMMENT "Building forward_vectorization.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_vectorization.o
  COMMAND cat ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/subgraph0.mlir |
  sed -e {s/@subgraph0/@subgraph0_vectorization/} |
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
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
  -batchmatmul-optimize
  -convert-linalg-to-affine-loops
  -lower-affine
  -convert-vector-to-scf
  -convert-scf-to-cf
  -llvm-request-c-wrappers
  -convert-vector-to-llvm
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
  -mattr=${BUDDY_OPT_ATTR} -filetype=obj
  -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Models/MobileNet-V3/subgraph0_vectorization.o
  DEPENDS ${BUDDY_BENCHMARK_DEEP_LEARNING_DIR}/Models/MobileNet-V3/subgraph0.mlir
  ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  COMMENT "Building subgraph0_vectorization.o"
  VERBATIM)

add_library(MOBILENETV3_AUTO_VECTORIZATION STATIC subgraph0_autoVectorization.o forward_autoVectorization.o)
set_target_properties(MOBILENETV3_AUTO_VECTORIZATION PROPERTIES LINKER_LANGUAGE CXX)

add_library(MOBILENETV3_VECTORIZATION STATIC subgraph0_vectorization.o forward_vectorization.o)
set_target_properties(MOBILENETV3_VECTORIZATION PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-model-mobileNetV3-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-model-mobileNetV3-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

if(CROSS_COMPILE_RVV)
  set(BUDDY_LIB_DIR ${BUDDY_MLIR_CROSS_LIB_DIR})
else()
  set(BUDDY_LIB_DIR ${BUDDY_MLIR_LIB_DIR})
endif()

target_link_libraries(dl-model-mobileNetV3-benchmark
  ${BenchmarkTool}
  MOBILENETV3_AUTO_VECTORIZATION
  MOBILENETV3_VECTORIZATION
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)
