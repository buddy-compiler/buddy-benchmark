# Buddy Benchmark

Buddy Benchmark is an extensible benchmark framework. 
We intend to provide a platform for performance comparison of various frameworks and optimizers.
This project is based on Google Benchmark. 

Clone the project:

```
$ git clone git@github.com:buddy-compiler/buddy-benchmark.git
```

## Image Processing Benchmark

Currently, the image processing benchmark includes the following frameworks or optimizers:

- OpenCV ([link](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html))

*NOTE: Please build OpenCV from source to achieve the best performance.*

- Buddy MLIR ([link](https://github.com/buddy-compiler/buddy-mlir))

*NOTE: Please make sure the `buddy-opt` tool of buddy-mlir project can work well.*

Run the image processing benchmark:

*Note: Please replace the `/PATH/TO/*` with your local path.
Use `-DBUDDY_OPT_STRIP_MINING` (default: 256) and `-DBUDDY_OPT_ATTR` (default: avx512f) to specify strip-mining size and architecture extension.*

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DIMAGE_PROCESSING_BENCHMARKS=ON \
    -DOpenCV_DIR=/PATH/TO/OPENCV/BUILD/ \
    -DBUDDY_OPT_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/
$ ninja image-processing-benchmark
$ cd bin && ./image-processing-benchmark
```

## Deep Learning Benchmark

*Note: Please replace the `/PATH/TO/*` with your local path.
Use `-DBUDDY_OPT_ATTR` (default: avx512f) to specify architecture extension.*

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DOpenCV_DIR=/PATH/TO/OPENCV/BUILD/ \
    -DBUDDY_OPT_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/
$ ninja
```

The deep learning benchmark includes the following e2e models and operations:

- MobileNet

We generated the model code with IREE and made appropriate modifications, and then compiled it with the MLIR tool chain.

Run the MobileNet benchmark:

```
$ cd <path to build>/bin && ./mobilenet-benchmark
```

- DepthwiseConv2DNhwcHwc Operation

Run the DepthwiseConv2DNhwcHwc operation benchmark:

```
$ cd <path to build>/bin && ./depthwise-conv-2d-nhwc-hwc-benchmark
```

## Testing

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DIMAGE_PROCESSING_BENCHMARKS=ON \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DBUILD_TESTS=ON \
    -DOpenCV_DIR=/path/to/opencv/build/ \
    -DBUDDY_OPT_BUILD_DIR=/path/to/buddy-mlir/build/ \
    -DBUDDY_OPT_STRIP_MINING=<strip mining size, default: 256> \
    -DBUDDY_OPT_ATTR=<ISA vector extension, default: avx512f>
$ cmake --build . --
$ ninja test
```

