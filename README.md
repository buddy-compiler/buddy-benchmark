# Buddy Benchmark

Buddy Benchmark is an extensible benchmark framework. 
We intend to provide a platform for performance comparison of various frameworks and optimizers.
This project is based on Google Benchmark. 

Clone the project:

```
$ git clone https://github.com/buddy-compiler/buddy-benchmark.git
```

## Image Processing Benchmark

Currently, the image processing benchmark includes the following frameworks or optimizers:

- OpenCV
- Buddy MLIR

NOTE: Please make sure the `conv-opt` tool of buddy-mlir project can work well.

Run the image processing benchmark:

```
$ cd buddy-benchmark
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DIMAGE_PROCESSING_BENCHMARKS=ON \
    -DOpenCV_DIR=/path/to/opencv/build/ \
    -DBUDDY_CONV_OPT_BUILD_DIR=/path/to/buddy-mlir/build \
    -DBUDDY_CONV_OPT_STRIP_MINING=<strip mining size, default: 256> \
    -DBUDDY_CONV_OPT_ATTR=<ISA vector extension, default: avx512f>
$ ninja image-processing-benchmark
$ cd bin
$ ./image-processing-benchmark
```

Note : The convolution implementation in buddy mlir is not feature complete at the moment and it may produce output which differs to some extent from the frameworks used in comparison. 
