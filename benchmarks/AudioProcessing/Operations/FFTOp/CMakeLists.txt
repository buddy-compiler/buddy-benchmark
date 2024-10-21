add_executable(dap-op-fft-benchmark
  KFRFft.cpp
  Main.cpp
)

target_link_directories(dap-op-fft-benchmark PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(dap-op-fft-benchmark PRIVATE
  # Third-party library
  kfr_io
  kfr_dft
  # Benchmark library
  GoogleBenchmark
)
