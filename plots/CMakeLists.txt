if(AUDIO_PROCESSING_BENCHMARKS)
  set(PYTHON_PATH "${PYTHON_BINARY_DIR}")
  set(MAIN_PATH "${PROJECT_SOURCE_DIR}")

  configure_file(source_dir.h.in source_dir.h @ONLY)

  include_directories(${CMAKE_CURRENT_BINARY_DIR})

  add_executable(audio-plot Main.cpp)
endif()