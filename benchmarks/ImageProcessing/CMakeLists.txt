#-------------------------------------------------------------------------------
# Optimization Configurations
#-------------------------------------------------------------------------------

if (${BUDDY_OPT_STRIP_MINING})
  set(SPLITING_SIZE ${BUDDY_OPT_STRIP_MINING} CACHE STRING "Spliting Size")
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256 CACHE STRING "Spliting Size")
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128 CACHE STRING "Spliting Size" FORCE)
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
elseif(HAVE_NEON)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
else()  # Fallback value, avoid crashing
  set(SPLITING_SIZE 16 CACHE STRING "Spliting Size")
endif()

set(MLIR_LINALG_TILE 2 CACHE STRING "MLIR Linalg Tile Size")
set(LLVM_MLIR_BINARY_DIR ${BUDDY_OPT_BUILD_DIR}/../llvm/build/bin)

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
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyConv2D.mlir 
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
#   COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
#           ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyConv2D.mlir 
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
# Buddy DIP Dialect Corr2D Operation + CB Algorithm
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT buddy-corr2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyCorr2D.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-corr2d.o
)
add_library(BuddyCorr2D STATIC buddy-corr2d.o)
set_target_properties(BuddyCorr2D PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT buddy-erosion2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyErosion2D.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-erosion2d.o
)
add_library(BuddyErosion2D STATIC buddy-erosion2d.o)
set_target_properties(BuddyErosion2D PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT buddy-dilation2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyDilation2D.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-dilation2d.o
)
add_library(BuddyDilation2D STATIC buddy-dilation2d.o)
set_target_properties(BuddyDilation2D PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT buddy-opening2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyOpening2D.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-opening2d.o
)
add_library(BuddyOpening2D STATIC buddy-opening2d.o)
set_target_properties(BuddyOpening2D PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT buddy-closing2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyClosing2D.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-closing2d.o
)
add_library(BuddyClosing2D STATIC buddy-closing2d.o)
set_target_properties(BuddyClosing2D PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT buddy-tophat2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyTopHat2D.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-tophat2d.o
)
add_library(BuddyTopHat2D STATIC buddy-tophat2d.o)
set_target_properties(BuddyTopHat2D PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT buddy-bottomhat2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyBottomHat2D.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-bottomhat2d.o
)
add_library(BuddyBottomHat2D STATIC buddy-bottomhat2d.o)
set_target_properties(BuddyBottomHat2D PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(OUTPUT buddy-morphgrad2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/BuddyMorphGrad2D.mlir 
            -lower-dip="DIP-strip-mining=${SPLITING_SIZE}" 
            -lower-affine -convert-scf-to-cf -convert-vector-to-llvm 
            --llvm-request-c-wrappers
            -convert-memref-to-llvm -convert-func-to-llvm 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/buddy-morphgrad2d.o
)
add_library(BuddyMorphGrad2D STATIC buddy-morphgrad2d.o)
set_target_properties(BuddyMorphGrad2D PROPERTIES LINKER_LANGUAGE CXX)

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
  BuddyErosion2DBenchmark.cpp
  BuddyDilation2DBenchmark.cpp
  BuddyOpening2DBenchmark.cpp
  BuddyClosing2DBenchmark.cpp
  BuddyTopHat2DBenchmark.cpp
  BuddyBottomHat2DBenchmark.cpp
  BuddyMorphGrad2DBenchmark.cpp
  OpenCVErode2DBenchmark.cpp
  OpenCVDilate2DBenchmark.cpp
  OpenCVOpening2DBenchmark.cpp
  OpenCVClosing2DBenchmark.cpp
  OpenCVTopHat2DBenchmark.cpp
  OpenCVBottomHat2DBenchmark.cpp
  OpenCVMorphGrad2DBenchmark.cpp
  )

target_include_directories(image-processing-benchmark 
  PRIVATE
  ${EIGEN_DIR}
  ${EIGEN_DIR}/unsupported/
)

target_link_libraries(image-processing-benchmark 
  GoogleBenchmark
  ${OpenCV_LIBS}
  MLIRConv2D
  BuddyConv2D
  BuddyCorr2D
  BuddyErosion2D
  BuddyDilation2D
  BuddyOpening2D
  BuddyClosing2D
  BuddyTopHat2D
  BuddyBottomHat2D
  BuddyMorphGrad2D
  Container
  PNGImage
  )
