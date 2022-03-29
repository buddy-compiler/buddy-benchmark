##===- buddy-benchmark.cmake ----------------------------------------------===//
##
## CMake helper functions.
##
##===----------------------------------------------------------------------===//
include(CMakeParseArguments)

# Build model benchmark
#
#  add_buddy_model_benchmark(name
#    [OpenCV, PNGImage]
#    MLIR    mlir file
#    BITCODE object file
#    LIBRARY library name
#    OPTIONS mlir-opt options
#    SRC     [src-1, src-2, ....]
#  )
function(add_buddy_model_benchmark name)

  # Parse arguments
  cmake_parse_arguments(ARGS
    "OpenCV;PNGImage"
    "MLIR;BITCODE;LIBRARY"
    "OPTIONS;SOURCE"
    ${ARGN}
  )

  # Compile MLIR file to LLVM bitcode
  add_custom_command(OUTPUT ${ARGS_BITCODE}
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_MLIR} 
      ${ARGS_OPTIONS} | ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
      ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} 
        --filetype=obj -o ${CMAKE_CURRENT_BINARY_DIR}/${ARGS_BITCODE}
  )

  add_library(${ARGS_LIBRARY} ${ARGS_BITCODE})

  set_target_properties(${ARGS_LIBRARY} PROPERTIES LINKER_LANGUAGE CXX)

  add_executable(${name} ${ARGS_SOURCE})

  # Link libraries
  target_link_directories(${name} PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
  target_link_libraries(${name}
    ${ARGS_LIBRARY}
    GoogleBenchmark
    Container
    mlir_c_runner_utils
  )
  if (${ARGS_OpenCV})
    target_link_libraries(${name} ${ARGS_LIBRARY} ${OpenCV_LIBS})
  endif()
  if (${ARGS_PNGImage})
    target_link_libraries(${name} ${ARGS_LIBRARY} PNGImage)
  endif()
endfunction()

# Build operation benchmark
#
#  add_ops_library_benchmark(name
#    MLIR     mlir file
#    BITCODE  llvm bitcode
#    LIBRARY  library name
#    OPTIONS   mlir-opt options
#    SRC      [src-1, [src-2, ...]]
#  )
function(add_buddy_ops_benchmark name)

  # Parse arguments
  cmake_parse_arguments(ARGS
    ""
    "MLIR;BITCODE;LIBRARY"
    "OPTIONS;SOURCE"
    ${ARGN}
  )

  # Compile MLIR file to LLVM bitcode
  add_custom_command(OUTPUT ${ARGS_BITCODE}
    COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt ${CMAKE_CURRENT_SOURCE_DIR}/${ARGS_MLIR}
    ${ARGS_OPTIONS} | ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir | 
    ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} -mattr=${BUDDY_OPT_ATTR} 
      --filetype=obj -o ${CMAKE_CURRENT_BINARY_DIR}/${ARGS_BITCODE}
  )

  add_library(${ARGS_LIBRARY} ${CMAKE_CURRENT_BINARY_DIR}/${ARGS_BITCODE})

  set_target_properties(${ARGS_LIBRARY} PROPERTIES LINKER_LANGUAGE CXX)

  add_executable(${name} ${ARGS_SOURCE})

  # Link libraries
  target_link_libraries(${name}
    ${ARGS_LIBRARY}
    GoogleBenchmark
    Container
  )
endfunction()
