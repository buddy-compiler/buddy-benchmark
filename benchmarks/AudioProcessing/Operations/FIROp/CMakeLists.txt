#-------------------------------------------------------------------------------
# Generate BuddyFIRTilesVectorization
#-------------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
function(build_buddy_tile_vectorization vector_size tile_size unroll_factor)
  add_custom_command(
    OUTPUT buddy_vec_${vector_size}_tile_${tile_size}_unroll_${unroll_factor}.o
    COMMAND
    cat ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/Operations/FIROp/FIR.mlir |
    sed -e 's/@buddy_fir_f32/@buddy_fir_vs_${vector_size}_ts_${tile_size}_uf_${unroll_factor}_f32/g'
        -e 's/@buddy_fir_f64/@buddy_fir_vs_${vector_size}_ts_${tile_size}_uf_${unroll_factor}_f64/g' |
      ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
        -vectorize-dap="fir-vec-size=${vector_size};fir-tile-size=${tile_size};fir-unroll-factor=${unroll_factor}"
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
        -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/Operations/FIROp/buddy_vec_${vector_size}_tile_${tile_size}_unroll_${unroll_factor}.o
    DEPENDS
      ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate
      ${LLVM_MLIR_BINARY_DIR}/llc 
  )
  add_library(BuddyVs${vector_size}Ts${tile_size}Uf${unroll_factor} STATIC buddy_vec_${vector_size}_tile_${tile_size}_unroll_${unroll_factor}.o)
  set_target_properties(BuddyVs${vector_size}Ts${tile_size}Uf${unroll_factor} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

build_buddy_tile_vectorization(8 64 1)
build_buddy_tile_vectorization(8 128 1)
build_buddy_tile_vectorization(8 256 1)
build_buddy_tile_vectorization(8 512 1)
build_buddy_tile_vectorization(8 1024 1)
build_buddy_tile_vectorization(8 2048 1)
build_buddy_tile_vectorization(8 4096 1)
build_buddy_tile_vectorization(8 8192 1)
build_buddy_tile_vectorization(16 64 1)
build_buddy_tile_vectorization(16 128 1)
build_buddy_tile_vectorization(16 216 1)
build_buddy_tile_vectorization(16 240 1)
build_buddy_tile_vectorization(16 256 1)
build_buddy_tile_vectorization(16 512 1)
build_buddy_tile_vectorization(16 512 2)
build_buddy_tile_vectorization(16 512 4)
build_buddy_tile_vectorization(16 512 8)
build_buddy_tile_vectorization(16 512 16)
build_buddy_tile_vectorization(16 1024 1)
build_buddy_tile_vectorization(16 2048 1)
build_buddy_tile_vectorization(16 4096 1)
build_buddy_tile_vectorization(16 8192 1)

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
  BuddyVs8Ts64Uf1
  BuddyVs8Ts128Uf1
  BuddyVs8Ts256Uf1
  BuddyVs8Ts512Uf1
  BuddyVs8Ts1024Uf1
  BuddyVs8Ts2048Uf1
  BuddyVs8Ts4096Uf1
  BuddyVs8Ts8192Uf1
  BuddyVs16Ts64Uf1
  BuddyVs16Ts128Uf1
  BuddyVs16Ts216Uf1
  BuddyVs16Ts240Uf1
  BuddyVs16Ts256Uf1
  BuddyVs16Ts512Uf1
  BuddyVs16Ts512Uf2
  BuddyVs16Ts512Uf4
  BuddyVs16Ts512Uf8
  BuddyVs16Ts512Uf16
  BuddyVs16Ts1024Uf1
  BuddyVs16Ts2048Uf1
  BuddyVs16Ts4096Uf1
  BuddyVs16Ts8192Uf1
  BuddyLibDAP
  # LLVM/MLIR library
  StaticMLIRCRunnerUtils
  # Benchmark library
  GoogleBenchmark
)
