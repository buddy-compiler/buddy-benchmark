#-------------------------------------------------------------------------------
# Audio Processing Benchmark Target
#-------------------------------------------------------------------------------

add_subdirectory(${KFR_DIR} ./kfr)

include_directories(${KFR_DIR}/include)

add_custom_command(OUTPUT buddy-fir.o
        COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
        ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/BuddyFir.mlir
        -lower-dap -convert-linalg-to-affine-loops
        -lower-affine -convert-scf-to-cf -convert-vector-to-llvm
        -convert-memref-to-llvm -convert-func-to-llvm='emit-c-wrappers=1'
        -reconcile-unrealized-casts |
        ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
        ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE}
        -mattr=${BUDDY_OPT_ATTR} --filetype=obj
        -o ${BUDDY_BINARY_DIR}/../benchmarks/AudioProcessing/buddy-fir.o
        )
add_library(BuddyFir STATIC buddy-fir.o)
set_target_properties(BuddyFir PROPERTIES LINKER_LANGUAGE CXX)

add_executable(audio-processing-benchmark
  KFRFft.cpp
  KFRFir.cpp 
  KFRBiquad.cpp
  KFRIir.cpp
  Main.cpp
  BuddyFirBenchmark.cpp
)

target_link_directories(audio-processing-benchmark PRIVATE ${KFR_DIR}/build/)
target_link_libraries(audio-processing-benchmark
  PRIVATE
  kfr_io
  kfr_dft
  BuddyFir
  GoogleBenchmark
)
