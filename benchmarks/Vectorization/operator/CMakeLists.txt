#-------------------------------------------------------------------------------
# Vectorization Benchmark Targets
#-------------------------------------------------------------------------------
add_subdirectory(MatMul)
add_subdirectory(MatVec)
add_subdirectory(Linalg_generic)
add_executable(vectorization-matrix-benchmark
  Main.cpp
  ./MatMul/MLIRMatMulBenchmark.cpp
  ./MatVec/MLIRMatVecBenchmark.cpp
  ./Linalg_generic/MLIRMiniLMBenchmark.cpp
  )


target_link_libraries(vectorization-matrix-benchmark
  GoogleBenchmark
  MLIRMatMul
  MLIRMatVec
  MiniLM_1
  MiniLM_2
  MiniLM_3
  MiniLM_4
  MiniLM_5
  MiniLM_6
  MiniLM_7
)

