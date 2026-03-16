
set(BUDDY_OPT ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate)
set(BUDDY_LLC ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-llc)
set(INTERFACES /home/xychen/buddy-mlir/frontend/Interfaces)

set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)

include_directories(
  ${BENCHMARKS_DIR}
  ${GEMMINI_INCLUDE_DIR}
  ${GEMMINI_INCLUDE_DIR}/../
  ${INTERFACES}
)

if (NOT DEFINED ENV{RISCV})
  message(FATAL_ERROR "Can't find RISCV environment variable(missing: RISCV_TOOLCHAIN)")
endif()

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT buddy_matmul_softmax.o
  COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmulSoftmax.mlir 
            -llvm-request-c-wrappers 
            -convert-linalg-to-gemmini 
            -convert-linalg-to-loops  
            -lower-gemmini | 
            ${BUDDY_TRANSLATE} -buddy-to-llvmir | 
            ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 
              -mattr=+buddyext,+D -float-abi=hard 
              -o buddy_matmul_softmax.o 
)
add_library(BuddyMatMulSoftmax STATIC buddy_matmul_softmax.o)
set_target_properties(BuddyMatMulSoftmax PROPERTIES LINKER_LANGUAGE C)

# add_library(ExoMatMul STATIC ExoMatmul.c)
# set_target_properties(ExoMatMul PROPERTIES LINKER_LANGUAGE C)

add_executable(dl-op-gemmini-matmul-softmax-benchmark Main.cpp)
target_link_libraries(dl-op-gemmini-matmul-softmax-benchmark
  -static
  # ExoMatMul
  BuddyMatMulSoftmax
)
