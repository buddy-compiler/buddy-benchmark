if(IMAGE_PROCESSING_BENCHMARKS)
  add_subdirectory(ImageProcessing)
endif()

if(DEEP_LEARNING_BENCHMARKS)
  add_subdirectory(DeepLearning)
endif()

if(AUDIO_PROCESSING_BENCHMARKS)
  add_subdirectory(AudioProcessing)
endif()

if(VECTORIZATION_BENCHMARKS)
  add_subdirectory(Vectorization)
endif()

if(OP_OPTIMIZATION_BENCHMARKS)
    add_subdirectory(OpOptimization)
endif()

if (GEMMINI)
    add_subdirectory(Gemmini)
endif()
