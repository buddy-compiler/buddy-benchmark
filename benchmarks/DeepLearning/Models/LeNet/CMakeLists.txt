add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
  COMMAND
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/buddy_lenet_import.py ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating forward.mlir, subgraph0.mlir"
)

add_custom_command(
  OUTPUT forward_auto_vectorization.o
  COMMAND 
    cat ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir |
    sed -e {s/@forward/@forward_auto_vectorization/}
        -e {s/@subgraph0/@subgraph0_auto_vectorization/} |
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
      -pass-pipeline
      "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
                                empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
                                func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
      -pass-pipeline
      "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
                                eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
                                convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
                                convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
                                convert-func-to-llvm, reconcile-unrealized-casts)" |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3
      -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} -filetype=obj
      -o ${CMAKE_CURRENT_BINARY_DIR}/forward_auto_vectorization.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  COMMENT "Building forward_auto_vectorization.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_auto_vectorization.o
  COMMAND
    cat ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_auto_vectorization/} |
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
    ${LLVM_MLIR_BINARY_DIR}/llc -O3
      -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} -filetype=obj
      -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_auto_vectorization.o
  DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  COMMENT "Building subgraph0_auto_vectorization.o"
  VERBATIM)

add_custom_command(
  OUTPUT forward_buddy_vectorization.o
  COMMAND
    cat ${CMAKE_CURRENT_SOURCE_DIR}/forward.mlir |
    sed -e {s/@forward/@forward_buddy_vectorization/}
        -e {s/@subgraph0/@subgraph0_buddy_vectorization/} |
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
      -pass-pipeline
      "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), \
                                empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg, arith-bufferize, \
                                func.func(linalg-bufferize, tensor-bufferize), func-bufferize)" |
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
      -pass-pipeline
      "builtin.module(func.func(buffer-deallocation-simplification, convert-linalg-to-loops), \
                                eliminate-empty-tensors, func.func(llvm-request-c-wrappers), \
                                convert-math-to-llvm, convert-math-to-libm, convert-scf-to-cf, \
                                convert-arith-to-llvm, expand-strided-metadata, finalize-memref-to-llvm, \
                                convert-func-to-llvm, reconcile-unrealized-casts)" |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3
      -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} -filetype=obj
      -o ${CMAKE_CURRENT_BINARY_DIR}/forward_buddy_vectorization.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  COMMENT "Building forward_buddy_vectorization.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0_buddy_vectorization.o
  COMMAND
    cat ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir |
    sed -e {s/@subgraph0/@subgraph0_buddy_vectorization/} |
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
    ${LLVM_MLIR_BINARY_DIR}/llc -O3
      -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} -filetype=obj
      -o ${CMAKE_CURRENT_BINARY_DIR}/subgraph0_buddy_vectorization.o
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph0.mlir
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt
  COMMENT "Building subgraph0_vectorization.o"
  VERBATIM)

add_library(LENET_AUTO_VECTORIZATION STATIC subgraph0_auto_vectorization.o forward_auto_vectorization.o)
set_target_properties(LENET_AUTO_VECTORIZATION PROPERTIES LINKER_LANGUAGE CXX)

add_library(LENET_BUDDY_VECTORIZATION STATIC subgraph0_buddy_vectorization.o forward_buddy_vectorization.o)
set_target_properties(LENET_BUDDY_VECTORIZATION PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-model-lenet-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-model-lenet-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

set(BenchmarkTool GoogleBenchmark)

if(CROSS_COMPILE_RVV)
  set(BUDDY_LIB_DIR ${BUDDY_MLIR_CROSS_LIB_DIR})
else()
  set(BUDDY_LIB_DIR ${BUDDY_MLIR_LIB_DIR})
endif()

target_link_libraries(dl-model-lenet-benchmark
  ${BenchmarkTool}
  LENET_AUTO_VECTORIZATION
  LENET_BUDDY_VECTORIZATION
  ${BUDDY_LIB_DIR}/libStaticMLIRCRunnerUtils.a
)
