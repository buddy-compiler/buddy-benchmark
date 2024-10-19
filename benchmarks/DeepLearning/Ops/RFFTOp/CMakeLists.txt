cmake_minimum_required(VERSION 3.10)
project(RFFTBenchmark)

add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_SOURCE_DIR}/result.txt
  COMMAND
    python3 ${CMAKE_CURRENT_SOURCE_DIR}/RFFT.py ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating result.txt"
)

add_custom_target(generate_result_txt ALL
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/result.txt
  COMMENT "Ensure result.txt is generated"
)

# Define variables for the cross-compilation toolchain and options.
if (CROSS_COMPILE_RVV)
  set(RISCV_GNU_TOOLCHAIN ${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain)
  set(RISCV_GNU_TOOLCHAIN_SYSROOT ${RISCV_GNU_TOOLCHAIN}/sysroot)
  set(BUDDY_OPT_TRIPLE riscv64)
  set(BUDDY_OPT_ATTR +v,+m)
endif()

# Add the executable for the benchmark
add_executable(dl-op-linalg-RFFT-benchmark
  GoogleBenchmarkMain.cpp
)

set_target_properties(dl-op-linalg-RFFT-benchmark PROPERTIES
  LINK_FLAGS "-static"
)

# Define the benchmark tool
set(BenchmarkTool GoogleBenchmark)

target_link_directories(dl-op-linalg-RFFT-benchmark PRIVATE ${BUDDY_MLIR_BUILD_DIR}/lib)

# Link the benchmark executable with the necessary libraries
target_link_libraries(dl-op-linalg-RFFT-benchmark
  PRIVATE
  ${BenchmarkTool}
  BuddyLibDAP
  ${BUDDY_MLIR_BUILD_DIR}/lib/libStaticMLIRCRunnerUtils.a
  GoogleBenchmark
)

add_dependencies(dl-op-linalg-RFFT-benchmark generate_result_txt)
