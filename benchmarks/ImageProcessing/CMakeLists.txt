set(BUDDY_OPT_STRIP_MINING 256 CACHE STRING "Stride Size")
set(LLVM_MLIR_BINARY_DIR ${BUDDY_OPT_BUILD_DIR}/../llvm/build/bin)

message(STATUS "Configuring Stride Size: ${BUDDY_OPT_STRIP_MINING}")

#-------------------------------------------------------------------------------
# MLIR Linalg Dialect Conv2D Operation + CB Algorithm
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT conv2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/Conv2D.mlir 
            -conv-vectorization="strip-mining=${BUDDY_OPT_STRIP_MINING}" 
            -lower-affine -convert-scf-to-std -convert-vector-to-llvm 
            -convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/conv2d.o
)
add_library(Conv2D STATIC conv2d.o)
set_target_properties(Conv2D PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Buddy DIP Dialect Corr2D Operation + CB Algorithm
#-------------------------------------------------------------------------------

add_custom_command(OUTPUT corr2d.o
  COMMAND ${BUDDY_OPT_BUILD_DIR}/bin/buddy-opt 
          ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/Corr2D.mlir 
            -lower-dip="DIP-strip-mining=${BUDDY_OPT_STRIP_MINING}" 
            -lower-affine -convert-scf-to-std -convert-vector-to-llvm 
            -convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' 
            -reconcile-unrealized-casts | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_OPT_TRIPLE} 
            -mattr=${BUDDY_OPT_ATTR} --filetype=obj 
            -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/corr2d.o
)
add_library(Corr2D STATIC corr2d.o)
set_target_properties(Corr2D PROPERTIES LINKER_LANGUAGE CXX)

#-------------------------------------------------------------------------------
# Image Processing Benchmark Target
#-------------------------------------------------------------------------------

add_executable(image-processing-benchmark
  Main.cpp 
  Filter2DBenchmark.cpp
  Conv2DBenchmark.cpp
  Corr2DBenchmark.cpp
  )

target_link_libraries(image-processing-benchmark 
  GoogleBenchmark
  ${OpenCV_LIBS}
  Conv2D
  Corr2D
  Container
  PNGImage
  )
