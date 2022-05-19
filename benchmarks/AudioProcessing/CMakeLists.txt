#-------------------------------------------------------------------------------
# Audio Processing Benchmark Target
#-------------------------------------------------------------------------------

add_subdirectory(${KFR_DIR} ./kfr)

include_directories(${KFR_DIR}/include)

add_executable(audio-processing-benchmark
  KFRFft.cpp
  KFRFir.cpp 
  KFRBiquad.cpp
  KFRIir.cpp
  Main.cpp
)

target_link_directories(audio-processing-benchmark PRIVATE ${KFR_DIR}/build/)
target_link_libraries(audio-processing-benchmark
  PRIVATE
  kfr_io
  kfr_dft
  GoogleBenchmark
)
