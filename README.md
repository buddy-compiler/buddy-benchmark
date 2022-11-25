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

- Eigen ([link](https://eigen.tuxfamily.org/index.php?title=Main_Page))

- Buddy MLIR ([link](https://github.com/buddy-compiler/buddy-mlir))

*NOTE: Please make sure the `buddy-opt` tool of buddy-mlir project can work well.*

Run the image processing benchmark:

| CMake Options  | Default Value |
| -------------- | ------------- |
| `-DBUDDY_OPT_STRIP_MINING`  | 256  |
| `-DMLIR_LINALG_TILE`  | 2  |
| `-DBUDDY_OPT_ATTR`  | avx512f  |
| `-DBUDDY_OPT_TRIPLE`  | x86_64-unknown-linux-gnu  |

*Note:*

*1. Please replace the `/PATH/TO/*` with your local path.*

*2. For running executable :*

*i. Please replace `<image path>` with path of the image which is to be used for*
*benchmarking.*

*ii. Please replace `<kernel name>` with name of the kernel which is to be used for*
*benchmarking as specifed in `include/ImageProcessing/Kernels.h`.*

*ii. Please replace `<kernelmorph name>` with name of the unsigned int kernel which is to be used for*
*benchmarking as specifed in `include/ImageProcessing/Kernels.h`.*

*iii. Please replace `<Boundary Option>` with `CONSTANT_PADDING` or `REPLICATE_PADDING`.*

Ex. `./image-processing-benchmark ../../benchmarks/ImageProcessing/Images/YuTu.png random3x3KernelAlign random3x3KernelAlignInt CONSTANT_PADDING`
```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DIMAGE_PROCESSING_BENCHMARKS=ON \
    -DOpenCV_DIR=/PATH/TO/OPENCV/BUILD/ \
    -DEIGEN_DIR=/PATH/TO/EIGEN/SOURCE/CODE \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/
$ ninja image-processing-benchmark
$ cd bin && ./image-processing-benchmark <image path> <kernel name> <kernelmorph name> <Boundary Option>
```

## Deep Learning Benchmark

| CMake Options  | Default Value |
| -------------- | ------------- |
| `-DBUDDY_OPT_ATTR`  | avx512f  |
| `-DBUDDY_OPT_TRIPLE`  | x86_64-unknown-linux-gnu  |

*Note: Please replace the `/PATH/TO/*` with your local path.*

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DOpenCV_DIR=/PATH/TO/OPENCV/BUILD/ \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/
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

## Audio Processing Benchmark

Currently, the audio processing benchmark includes the following frameworks or optimizers:

- KFR ([link](https://github.com/kfrlib/kfr))

*Note: Please replace the `/PATH/TO/*` with your local path.*

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DAUDIO_PROCESSING_BENCHMARKS=ON \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DKFR_DIR=/PATH/TO/KFR/SOURCE/CODE \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/
$ ninja audio-processing-benchmark
$ cd bin
$ ./audio-processing-benchmark
```

### audio-plot tool

To better demonstrate the result after processing, we provide a tool for figure plotting. To use this tool, you have to make sure that you are using `python3` and that the `numpy`, `matplotlib` and `scipy` packages have been installed properly. Use the following command to install the required packages:

```
$ pip install matplotlib scipy
```

You can customize the `python3` path by adding the option `-DPYTHON_BINARY_DIR=/PATH/TO/PYTHON/BIN` while building:

*Note: Please replace the `/PATH/TO/*` with your local path.*

```
$ cd build
$ cmake -G Ninja .. \
    -DAUDIO_PROCESSING_BENCHMARKS=ON \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DKFR_DIR=/PATH/TO/KFR/SOURCE/CODE \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD \
    -DPYTHON_BINARY_DIR=/PATH/TO/PYTHON/BIN/
$ ninja audio-plot
```

Once the processing is done, you can use this tool to plot a comparision figure:

```
$ cd bin
$ ./audio-plot ../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav ResultKFRIir.wav
```

The result is saved in `bin/res.png`. For more usage, use `audio-plot -h` for detailed information.

## Vectorization Benchmark

*Note: Please replace the `/PATH/TO/*` with your local path.*

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DVECTORIZATION_BENCHMARKS=ON \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/
$ ninja vectorization-benchmark
$ cd bin
$ ./vectorization-benchmark
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
    -DBUDDY_MLIR_BUILD_DIR=/path/to/buddy-mlir/build/ \
    -DBUDDY_OPT_STRIP_MINING=<strip mining size, default: 256> \
    -DBUDDY_OPT_ATTR=<ISA vector extension, default: avx512f>
$ cmake --build . --
$ ninja test
```

