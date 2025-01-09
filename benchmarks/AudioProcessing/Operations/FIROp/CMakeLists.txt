#-------------------------------------------------------------------------------
# Generate MLIRFIRVectorization
#-------------------------------------------------------------------------------

function(build_fir_vectorization type)
  add_custom_command(
    OUTPUT fir-vectorization-${type}.o
    COMMAND
      cat ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/FIROp/MLIRFIRVectorization.mlir |
      sed 's/TYPE_PLACEHOLDER/${type}/g' |
      ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        -convert-scf-to-cf
        -convert-vector-to-llvm
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc 
        -mtriple=${BUDDY_OPT_TRIPLE} 
        -mattr=${BUDDY_OPT_ATTR} 
        -filetype=obj 
        -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/FIROp/fir-vectorization-${type}.o
    DEPENDS
      ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate
      ${LLVM_MLIR_BINARY_DIR}/llc 
  )
  add_library(MLIRFIRVectorization${type} STATIC fir-vectorization-${type}.o)
  set_target_properties(MLIRFIRVectorization${type} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_fir_vectorization(f32)
build_fir_vectorization(f64)

#-------------------------------------------------------------------------------
# Generate dap-op-fir-benchmark
#-------------------------------------------------------------------------------

add_executable(dap-op-fir-benchmark Main.cpp)

target_link_directories(dap-op-fir-benchmark PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(dap-op-fir-benchmark PRIVATE
  # Third-party library
  kfr_io
  # MLIR hand-written benchmark
  MLIRFIRVectorizationf32
  MLIRFIRVectorizationf64
  # Buddy DAP library
  BuddyLibDAP
  # LLVM/MLIR library
  StaticMLIRCRunnerUtils
  # Benchmark library
  GoogleBenchmark
)
