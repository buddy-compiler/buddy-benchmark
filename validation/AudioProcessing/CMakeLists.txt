#-------------------------------------------------------------------------------
# Audio Processing Correctness Target
#-------------------------------------------------------------------------------
if (NOT AUDIO_PROCESSING_BENCHMARKS)
  add_subdirectory(${KFR_DIR} ./kfr)
  include_directories(${KFR_DIR}/include)
endif()

add_library(AudioValidationLib SHARED AudioValidationLib.cpp)

target_link_directories(AudioValidationLib
  PRIVATE 
  ${KFR_DIR}/build/
  ${BUDDY_MLIR_LIB_DIR}
)

target_link_libraries(AudioValidationLib
  PRIVATE
  kfr_io
  kfr_dft
  BuddyLibDAP
)