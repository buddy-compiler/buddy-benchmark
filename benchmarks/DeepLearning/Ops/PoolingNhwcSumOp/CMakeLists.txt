#-------------------------------------------------------------------------------
# MLIR Linalg Dialect PoolingNhwcSum Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-pooling-nhwc-sum.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${CMAKE_CURRENT_SOURCE_DIR}/MLIRPoolingNhwcSum.mlir 
            -convert-linalg-to-loops
            -convert-scf-to-cf 
            -convert-linalg-to-llvm 
            -lower-affine 
            --convert-memref-to-llvm 
            -convert-func-to-llvm='emit-c-wrappers=1' 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj -o 
          ${CMAKE_CURRENT_BINARY_DIR}/mlir-pooling-nhwc-sum.o
)
add_library(MLIRPoolingNhwcSum ${CMAKE_CURRENT_BINARY_DIR}/mlir-pooling-nhwc-sum.o)
set_target_properties(MLIRPoolingNhwcSum PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR Linalg Dialect PoolingNhwcSum Operation + CB Pooling algorithm
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT buddy-pooling-nhwc-sum.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
          ${CMAKE_CURRENT_SOURCE_DIR}/BuddyPoolingNhwcSum.mlir 
            -pooling-vectorization 
            -lower-affine 
            -convert-vector-to-scf 
            -convert-scf-to-cf 
            -tensor-bufferize 
            -convert-vector-to-llvm 
            -convert-memref-to-llvm 
            -convert-arith-to-llvm 
            -convert-func-to-llvm='emit-c-wrappers=1' 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj -o 
          ${CMAKE_CURRENT_BINARY_DIR}/buddy-pooling-nhwc-sum.o
)
add_library(BuddyPoolingNhwcSum ${CMAKE_CURRENT_BINARY_DIR}/buddy-pooling-nhwc-sum.o)
set_target_properties(BuddyPoolingNhwcSum PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# PoolingNhwcSum Benchmark Target
#-------------------------------------------------------------------------------

add_executable(pooling-nhwc-sum-benchmark
  Main.cpp 
  MLIRPoolingNhwcSumBenchmark.cpp
  BuddyPoolingNhwcSumBenchmark.cpp
  )

target_link_libraries(pooling-nhwc-sum-benchmark 
  GoogleBenchmark
  MLIRPoolingNhwcSum
  BuddyPoolingNhwcSum
  Container
  )
