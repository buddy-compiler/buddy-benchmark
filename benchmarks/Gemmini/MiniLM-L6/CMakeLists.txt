set(BUDDY_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/bin)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
set(GEMMINI_CONFIG "dim=4 acc_rows=4096 acc_t=f32 elem_t=f32")
set(MINILM-L6-DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../DeepLearning/Models/MiniLM-L6)

if (NOT DEFINED ENV{RISCV})
  message(FATAL_ERROR "Can't find RISCV environment variable(missing: RISCV_TOOLCHAIN)")
endif()

add_custom_command(
  OUTPUT 
    minilm-l6.o
  COMMAND 
    ${BUDDY_MLIR_BINARY_DIR}/buddy-opt ${MINILM-L6-DIR}/MiniLM-L6-12.mlir
    --linalg-bufferize
    --llvm-request-c-wrappers
    --convert-linalg-to-gemmini="acc_t=f32"
    --convert-linalg-to-loops
    --func-bufferize
    --arith-bufferize
    --tensor-bufferize
    --finalizing-bufferize
    --lower-affine
    --convert-scf-to-cf
    --expand-strided-metadata
    --convert-vector-to-llvm
    --memref-expand
    --arith-expand
    -lower-gemmini=${GEMMINI_CONFIG}
    --convert-arith-to-llvm
    --finalize-memref-to-llvm
    --convert-math-to-llvm
    -reconcile-unrealized-casts | 
    ${BUDDY_MLIR_BINARY_DIR}/buddy-translate -buddy-to-llvmir |
    ${BUDDY_MLIR_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 
    -mattr=+buddyext,+D -float-abi=hard -o minilm-l6.o
)

add_library(MINILM-L6 STATIC minilm-l6.o)
set_target_properties(MINILM-L6 PROPERTIES LINKER_LANGUAGE CXX)
add_executable(Gemmini-MiniLM-L6 Main.cpp)
target_link_libraries(Gemmini-MiniLM-L6
  -static
  MINILM-L6
  CRunnerUtils
)
