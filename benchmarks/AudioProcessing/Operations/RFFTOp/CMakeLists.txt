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

# Add the executable for the benchmark
add_executable(dap-op-rfft-benchmark
  GoogleBenchmarkMain.cpp
)

# Define the benchmark tool
set(BenchmarkTool GoogleBenchmark)

target_link_directories(dap-op-rfft-benchmark PRIVATE 
  ${BUDDY_MLIR_LIB_DIR}
  ${LLVM_MLIR_LIBRARY_DIR}
)

# Link the benchmark executable with the necessary libraries
target_link_libraries(dap-op-rfft-benchmark PRIVATE
  ${BenchmarkTool}
  BuddyLibDAP
  mlir_c_runner_utils
  GoogleBenchmark
)

add_dependencies(dap-op-rfft-benchmark generate_result_txt)
