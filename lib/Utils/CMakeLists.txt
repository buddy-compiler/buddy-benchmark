if(DEFINED IMAGE_PROCESSING_BENCHMARKS OR DEEP_LEARNING_BENCHMARKS)
  add_library(Container Container.cpp)
  add_library(PNGImage PNGImage.cpp)

  target_link_libraries(Container ${OpenCV_LIBS})
  target_link_libraries(PNGImage ${PNG_LIBRARIES})
endif()
