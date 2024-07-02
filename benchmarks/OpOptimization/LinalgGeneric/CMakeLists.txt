add_custom_command(OUTPUT mlir-generic-origin.o
  COMMAND 
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt  ${CMAKE_CURRENT_SOURCE_DIR}/linalg_generic_origin.mlir
      # -convert-linalg-to-affine-loops
      # -lower-affine
      # -convert-vector-to-llvm
      # -finalize-memref-to-llvm
      # -convert-scf-to-cf
      # -convert-linalg-to-llvm
      # -llvm-request-c-wrappers
      # -convert-func-to-llvm
      # -one-shot-bufferize='bufferize-function-boundaries'
      -llvm-request-c-wrappers
      -test-lower-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
      -filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/LinalgGeneric/mlir-generic-origin.o
)

add_library(GenericOrigin STATIC mlir-generic-origin.o)
set_target_properties(GenericOrigin PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT mlir-transform-tiling.o
  COMMAND 
    cat ${CMAKE_CURRENT_SOURCE_DIR}/linalg_generic_origin.mlir |
    sed 's/origin/transform_tiling/' |
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      -polyhedral-tiling='tile-sizes=32,32'
      # -linalg-bufferize
      # -convert-linalg-to-loops
      # -convert-vector-to-llvm
      # -lower-affine
      # -convert-scf-to-cf
      # -func-bufferize
      # -finalizing-bufferize
      # -finalize-memref-to-llvm
      # -llvm-request-c-wrappers
      # -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
      -one-shot-bufferize='bufferize-function-boundaries'
      -llvm-request-c-wrappers
      -test-lower-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
      -filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/LinalgGeneric/mlir-transform-tiling.o
)

add_library(TransformTiling STATIC mlir-transform-tiling.o)
set_target_properties(TransformTiling PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT mlir-manual-opt.o
  COMMAND 
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt  ${CMAKE_CURRENT_SOURCE_DIR}/linalg_manual_opt.mlir
    #  -one-shot-bufferize='bufferize-function-boundaries'
     -llvm-request-c-wrappers
     -test-lower-to-llvm
       |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
      -filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/LinalgGeneric/mlir-manual-opt.o
)

add_library(ManualOpt STATIC mlir-manual-opt.o)
set_target_properties(ManualOpt PROPERTIES LINKER_LANGUAGE CXX)

add_executable(linalg-generic-benchmark
  Main.cpp
  GenericBenchmark.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
target_link_directories(linalg-generic-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(linalg-generic-benchmark
  GoogleBenchmark
  GenericOrigin
  TransformTiling
  ManualOpt
  mlir_c_runner_utils
  mlir_runner_utils
)