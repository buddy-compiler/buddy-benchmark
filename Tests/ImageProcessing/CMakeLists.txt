add_executable(Tests TestImageGenerator.cpp)

target_link_libraries(Tests
  ${OpenCV_LIBS}
  )
