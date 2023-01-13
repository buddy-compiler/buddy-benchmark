#-------------------------------------------------------------------------------
# Optimization Configurations
#-------------------------------------------------------------------------------

if (${BUDDY_OPT_STRIP_MINING})
  set(SPLITING_SIZE ${BUDDY_OPT_STRIP_MINING} CACHE STRING "Spliting Size")
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256 CACHE STRING "Spliting Size")
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128 CACHE STRING "Spliting Size")
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
elseif(HAVE_NEON)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
else()  # Fallback value, avoid crashing
  set(SPLITING_SIZE 16 CACHE STRING "Spliting Size")
endif()

#-------------------------------------------------------------------------------
# Buddy DIP Dialect
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT dip.o
  COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_MLIR_BUILD_DIR}/../examples/DIPDialect/dip.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" -arith-expand 
            -lower-affine -convert-scf-to-cf -convert-math-to-llvm 
            -convert-vector-to-llvm -convert-memref-to-llvm 
            -llvm-request-c-wrappers -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/DeepLearning/Models/dip.o
)
add_library(DIP STATIC dip.o)
set_target_properties(DIP PROPERTIES LINKER_LANGUAGE CXX)

add_buddy_model_benchmark(mobilenet-v3-benchmark
  OpenCV
  DIP
  MLIR MobileNetV3.mlir
  BITCODE mobilenet-v3-default.o
  LIBRARY MobileNetV3Default
  OPTIONS
    --test-linalg-transform-patterns="test-generalize-pad-tensor"
    --linalg-bufferize 
    --convert-linalg-to-loops 
    --func-bufferize 
    --arith-bufferize 
    --tensor-bufferize 
    --finalizing-bufferize 
    --convert-vector-to-scf 
    --convert-scf-to-cf 
    --expand-strided-metadata
    --lower-affine 
    --convert-vector-to-llvm 
    -memref-expand -arith-expand 
    --convert-arith-to-llvm
    --convert-memref-to-llvm 
    --convert-math-to-llvm
    --llvm-request-c-wrappers
    --convert-func-to-llvm
    --reconcile-unrealized-casts
  SOURCE Main.cpp MobileNetBenchmark.cpp
)
