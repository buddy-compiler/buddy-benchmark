
set(BUDDY_OPT ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt)
set(BUDDY_TRANSLATE ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-translate)
set(BUDDY_LLC ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-llc)
set(INTERFACES /home/xychen/buddy-mlir/frontend/Interfaces)

set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)


if (NOT DEFINED ENV{RISCV})
  message(FATAL_ERROR "Can't find RISCV environment variable(missing: RISCV_TOOLCHAIN)")
endif()

# CMAKE_C_FLAGS is set when configuring cmake.
separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

add_custom_command(OUTPUT matmul.o
  COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir 
            -llvm-request-c-wrappers 
            -convert-linalg-to-gemmini 
            -convert-linalg-to-loops  
            -lower-gemmini | 
            ${BUDDY_TRANSLATE} -buddy-to-llvmir | 
            ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 
              -mattr=+buddyext,+D -float-abi=hard 
              -o matmul.o 
)
add_library(GemminiMatMul STATIC matmul.o)
set_target_properties(GemminiMatMul PROPERTIES LINKER_LANGUAGE CXX)

add_library(Utils STATIC Utils.hpp)
set_target_properties(Utils PROPERTIES LINKER_LANGUAGE CXX)

add_executable(dl-op-gemmini-matmul-benchmark Main.cpp)

target_include_directories(dl-op-gemmini-matmul-benchmark
  PRIVATE
  ${INTERFACES}
)

target_link_libraries(dl-op-gemmini-matmul-benchmark
  -static
  GemminiMatMul
  Utils
)


# Build the target for your new method here.
