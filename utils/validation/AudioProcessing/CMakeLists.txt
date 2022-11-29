#-------------------------------------------------------------------------------
# Audio Processing Correctness Target
#-------------------------------------------------------------------------------
if (NOT AUDIO_PROCESSING_BENCHMARKS)
  add_subdirectory(${KFR_DIR} ./kfr)
  include_directories(${KFR_DIR}/include)
endif()

add_custom_command(OUTPUT buddy-fir.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
  ${BUDDY_SOURCE_DIR}/benchmarks/AudioProcessing/BuddyFir.mlir
  -lower-dap -convert-linalg-to-affine-loops
  -lower-affine -convert-scf-to-cf -convert-vector-to-llvm
  --llvm-request-c-wrappers
  -convert-memref-to-llvm -convert-func-to-llvm
  -reconcile-unrealized-casts |
  ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
  ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE}
  -mattr=${BUDDY_OPT_ATTR} --filetype=obj
  -o ${BUDDY_BINARY_DIR}/../Utils/validation/AudioProcessing/buddy-fir.o
  )
add_library(BuddyFirCor SHARED buddy-fir.o)

set_target_properties(BuddyFirCor PROPERTIES LINKER_LANGUAGE CXX)

add_library(CWrapper SHARED CWrapper.cpp CWrapper.h buddy-fir.o)
