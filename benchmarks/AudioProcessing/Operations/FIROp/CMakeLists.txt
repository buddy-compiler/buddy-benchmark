#-------------------------------------------------------------------------------
# Generate BuddyFIRTilesVectorization
#-------------------------------------------------------------------------------

function(build_buddy_tile_vectorization vector_size tile_size)
  add_custom_command(
    OUTPUT buddy_tile_${vector_size}_vectorization_${tile_size}.o
    COMMAND
    cat ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/FIROp/FIR.mlir |
    sed -e 's/@buddy_fir_f32/@buddy_fir_vs_${vector_size}_ts_${tile_size}_f32/g'
        -e 's/@buddy_fir_f64/@buddy_fir_vs_${vector_size}_ts_${tile_size}_f64/g' |
      ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        -vectorize-dap="fir-vec-size=${vector_size};fir-tile-size=${tile_size}"
        -convert-scf-to-cf
        -convert-vector-to-llvm
        -llvm-request-c-wrappers
        -convert-arith-to-llvm
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate -mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -O3
        -mtriple=${BUDDY_OPT_TRIPLE} 
        -mattr=${BUDDY_OPT_ATTR} 
        -filetype=obj 
        -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/FIROp/buddy_tile_${vector_size}_vectorization_${tile_size}.o
    DEPENDS
      ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate
      ${LLVM_MLIR_BINARY_DIR}/llc 
  )
  add_library(BuddyVs${vector_size}Ts${tile_size} STATIC buddy_tile_${vector_size}_vectorization_${tile_size}.o)
  set_target_properties(BuddyVs${vector_size}Ts${tile_size} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_buddy_tile_vectorization(8 64)
build_buddy_tile_vectorization(8 128)
build_buddy_tile_vectorization(8 256)
build_buddy_tile_vectorization(8 512)
build_buddy_tile_vectorization(8 1024)
build_buddy_tile_vectorization(8 2048)
build_buddy_tile_vectorization(8 4096)
build_buddy_tile_vectorization(8 8192)
build_buddy_tile_vectorization(16 64)
build_buddy_tile_vectorization(16 128)
build_buddy_tile_vectorization(16 256)
build_buddy_tile_vectorization(16 512)
build_buddy_tile_vectorization(16 1024)
build_buddy_tile_vectorization(16 2048)
build_buddy_tile_vectorization(16 4096)
build_buddy_tile_vectorization(16 8192)

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
  BuddyVs8Ts64
  BuddyVs8Ts128
  BuddyVs8Ts256
  BuddyVs8Ts512
  BuddyVs8Ts1024
  BuddyVs8Ts2048
  BuddyVs8Ts4096
  BuddyVs8Ts8192
  BuddyVs16Ts64
  BuddyVs16Ts128
  BuddyVs16Ts240
  BuddyVs16Ts256
  BuddyVs16Ts512
  BuddyVs16Ts1024
  BuddyVs16Ts2048
  BuddyVs16Ts4096
  BuddyVs16Ts8192
  BuddyLibDAP
  # LLVM/MLIR library
  StaticMLIRCRunnerUtils
  # Benchmark library
  GoogleBenchmark
)
