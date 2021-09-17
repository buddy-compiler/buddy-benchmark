set(BUDDY_CONV_OPT_STRIP_MINING 256)
set(BUDDY_CONV_OPT_ATTR avx512f)
set(LLVM_MLIR_BINARY_DIR ${BUDDY_CONV_OPT_BUILD_DIR}/../llvm/build/bin)

add_custom_command(OUTPUT conv2d.o
  COMMAND ${BUDDY_CONV_OPT_BUILD_DIR}/bin/conv-opt ${BUDDY_SOURCE_DIR}/benchmarks/ImageProcessing/Conv2D.mlir -conv-vectorization="strip-mining=${BUDDY_CONV_OPT_STRIP_MINING}" -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm='emit-c-wrappers=1' | 
          ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
          ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=${BUDDY_CONV_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../benchmarks/ImageProcessing/conv2d.o
)

add_library(Conv2D STATIC conv2d.o)

set_target_properties(Conv2D PROPERTIES LINKER_LANGUAGE CXX)

add_executable(image-processing-benchmark Main.cpp OpenCVBenchmark.cpp BuddyBenchmark.cpp)

target_link_libraries(image-processing-benchmark 
  GoogleBenchmark
  ${OpenCV_LIBS}
  Conv2D
  BuddyContainer
  )
