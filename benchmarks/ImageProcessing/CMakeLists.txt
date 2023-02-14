#-------------------------------------------------------------------------------
# Optimization Configurations
#-------------------------------------------------------------------------------

if (${BUDDY_OPT_STRIP_MINING})
  set(SPLITING_SIZE ${BUDDY_OPT_STRIP_MINING} CACHE STRING "Spliting Size")
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256 CACHE STRING "Spliting Size")
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128 CACHE STRING "Spliting Size")
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
elseif(HAVE_NEON)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
else()  # Fallback value, avoid crashing
  set(SPLITING_SIZE 16 CACHE STRING "Spliting Size")
endif()

set(MLIR_LINALG_TILE 2 CACHE STRING "MLIR Linalg Tile Size")
set(LLVM_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin)

message(STATUS "Configuring Stride Size: ${SPLITING_SIZE}")
message(STATUS "Configuring MLIR Linalg Tile Size: ${MLIR_LINALG_TILE}")

#-------------------------------------------------------------------------------
# MLIR Linalg Dialect Conv2D Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT mlir-conv2d.o
  COMMAND ${LLVM_MLIR_BINARY_DIR}/mlir-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/MLIRConv2D.mlir 
            -convert-linalg-to-loops
            -convert-scf-to-cf -convert-linalg-to-llvm
            -lower-affine -convert-scf-to-cf --convert-memref-to-llvm 
            --llvm-request-c-wrappers
            -convert-func-to-llvm
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/mlir-conv2d.o
)
add_library(MLIRConv2D STATIC mlir-conv2d.o)
set_target_properties(MLIRConv2D PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# MLIR Linalg Dialect Conv2D Operation + Buddy CB Algorithm
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT buddy-conv2d.o
  COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
          ${BUDDY_MLIR_BUILD_DIR}/../examples/ConvOpt/conv2d.mlir 
            -conv-vectorization="strip-mining=${SPLITING_SIZE}" 
            -convert-vector-to-scf 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-conv2d.o
)
add_library(BuddyConv2D STATIC buddy-conv2d.o)
set_target_properties(BuddyConv2D PROPERTIES LINKER_LANGUAGE CXX)

# add_custom_command(OUTPUT buddy-conv2d.o
#   COMMAND ${BUDDY_MLIR_BUILD_DIR}/bin/buddy-opt
#           ${BUDDY_MLIR_BUILD_DIR}/../examples/ConvOpt/conv2d.mlir 
#             -linalg-tile="tile-sizes=${MLIR_LINALG_TILE},${MLIR_LINALG_TILE}"
#             -conv-vectorization="tile-sizes=${MLIR_LINALG_TILE},${MLIR_LINALG_TILE}" 
#             -convert-vector-to-scf 
#             -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
#             --llvm-request-c-wrappers
#             -convert-memref-to-llvm -convert-func-to-llvm 
#             -reconcile-unrealized-casts | 
#           ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
#           ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
#             -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
#             -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-conv2d.o
# )
# add_library(BuddyConv2D STATIC buddy-conv2d.o)
# set_target_properties(BuddyConv2D PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Image Processing Benchmark Target
#-------------------------------------------------------------------------------

add_executable(image-processing-benchmark
  Main.cpp 
  OpenCVFilter2DBenchmark.cpp
  EigenConvolve2DBenchmark.cpp
  MLIRConv2DBenchmark.cpp
  BuddyConv2DBenchmark.cpp
  BuddyCorr2DBenchmark.cpp
  BuddyMorph2DBenchmark.cpp
  OpenCVMorph2DBenchmark.cpp
  )

target_include_directories(image-processing-benchmark 
  PRIVATE
  ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/include/
  ${EIGEN_DIR}
  ${EIGEN_DIR}/unsupported/
  )

target_link_directories(image-processing-benchmark
  PRIVATE 
  ${BUDDY_MLIR_LIB_DIR}
  )

target_link_libraries(image-processing-benchmark 
  GoogleBenchmark
  ${OpenCV_LIBS}
  MLIRConv2D
  BuddyConv2D
  # Link Buddy MLIR DIP Library.
  BuddyLibDIP
  )
