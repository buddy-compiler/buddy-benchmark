
set(BUDDY_OPT ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate)
set(BUDDY_LLC ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-llc)
set(INTERFACES ${BUDDY_MLIR_BUILD_DIR}/../frontend/Interfaces)

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

add_custom_command(
  OUTPUT buddy_conv.o
  COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/conv.mlir 
    -llvm-request-c-wrappers 
    -lower-gemmini
    > log.mlir
  COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/conv.mlir 
            -llvm-request-c-wrappers 
            -convert-linalg-to-gemmini 
            -convert-linalg-to-loops  
            -lower-gemmini | 
            ${BUDDY_TRANSLATE} -buddy-to-llvmir | 
            ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 
              -mattr=+buddyext,+D -float-abi=hard 
              -o buddy_conv.o 
    VERBATIM)

add_library(BuddyConv STATIC buddy_conv.o)
set_target_properties(BuddyConv PROPERTIES LINKER_LANGUAGE C)

add_library(ExoConv STATIC ExoConv.c)
set_target_properties(ExoConv PROPERTIES LINKER_LANGUAGE C)

add_executable(dl-op-gemmini-conv-benchmark Main.cpp)
target_link_libraries(dl-op-gemmini-conv-benchmark
  -static
  ExoConv
  BuddyConv
)
