add_custom_command(OUTPUT mlir-llama-origin.o
  COMMAND 
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt  ${CMAKE_CURRENT_SOURCE_DIR}/llama_origin.mlir
      # -convert-linalg-to-affine-loops
      # -lower-affine
      # -convert-vector-to-llvm
      # -finalize-memref-to-llvm
      # -convert-scf-to-cf
      # -convert-linalg-to-llvm
      # -llvm-request-c-wrappers
      # -convert-func-to-llvm
      -one-shot-bufferize='bufferize-function-boundaries'
      -llvm-request-c-wrappers
      -test-lower-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
      -filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/LLama/mlir-llama-origin.o
)

add_library(LLamaOrigin STATIC mlir-llama-origin.o)
set_target_properties(LLamaOrigin PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT mlir-llama-tiling.o
  COMMAND 
    cat ${CMAKE_CURRENT_SOURCE_DIR}/llama_origin.mlir |
    sed 's/base/tiling/' |
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      -polyhedral-tiling='tile-sizes=32,32,32,32'
      # -generic-vectorization=tile-sizes=0,32,32,32
      # -convert-vector-to-scf
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
    # ${LLVM_MLIR_BINARY_DIR}/mlir-opt 
    /home/heyi/heyi/llvm-b1115f8c-ubuntu-x64/bin/mlir-opt
      -convert-linalg-to-affine-loops
      -affine-super-vectorize='virtual-vector-size=16'
      -one-shot-bufferize='bufferize-function-boundaries'
      -llvm-request-c-wrappers
      -test-lower-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
      -filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/LLama/mlir-llama-tiling.o
)

add_library(LLamaTiling STATIC mlir-llama-tiling.o)
set_target_properties(LLamaTiling PROPERTIES LINKER_LANGUAGE CXX)


add_custom_command(OUTPUT mlir-affine-vec.o
  COMMAND 
    cat ${CMAKE_CURRENT_SOURCE_DIR}/llama_origin.mlir |
    sed 's/base/affine_vec/' |
    ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      -convert-linalg-to-affine-loops
      # -affine-loop-tile='tile-sizes=32,32,32,32'
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
      # -vectorize-affine-loop-nest
      -affine-super-vectorize='virtual-vector-size=128' 
      -one-shot-bufferize='bufferize-function-boundaries'
      -llvm-request-c-wrappers
      -test-lower-to-llvm
      -reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
      -filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/LLama/mlir-affine-vec.o
)

add_library(AffineVec STATIC mlir-affine-vec.o)
set_target_properties(AffineVec PROPERTIES LINKER_LANGUAGE CXX)

# add_custom_command(OUTPUT mlir-manual-opt.o
#   COMMAND 
#     ${LLVM_MLIR_BINARY_DIR}/mlir-opt  ${CMAKE_CURRENT_SOURCE_DIR}/linalg_manual_opt.mlir
#     #  -one-shot-bufferize='bufferize-function-boundaries'
#      -llvm-request-c-wrappers
#      -test-lower-to-llvm
#        |
#     ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
#     ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR}
#       -filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/OpOptimization/LinalgGeneric/mlir-manual-opt.o
# )

# add_library(ManualOpt STATIC mlir-manual-opt.o)
# set_target_properties(ManualOpt PROPERTIES LINKER_LANGUAGE CXX)

add_executable(llama-benchmark
  Main.cpp
  LLamaBenchmark.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
target_link_directories(llama-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(llama-benchmark
  GoogleBenchmark
  LLamaOrigin
  LLamaTiling
  AffineVec
  # ManualOpt
  mlir_c_runner_utils
  mlir_runner_utils
)