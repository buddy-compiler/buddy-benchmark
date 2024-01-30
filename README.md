# Buddy Benchmark

Buddy Benchmark is an extensible benchmark framework. 
We intend to provide a platform for performance comparison of various frameworks and optimizers.
This project is based on Google Benchmark. 

Clone the project:

```
$ git clone git@github.com:buddy-compiler/buddy-benchmark.git
```

## Choose and Build Dependencies

###  Choose Submodules

```
$ git submodule update --init
```

### Build OpenCV

```
$ cd buddy-benchmark/thirdparty/opencv
$ mkdir build && cd build
$ cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
$ ninja
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
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DIMAGE_PROCESSING_BENCHMARKS=ON \
    -DOpenCV_DIR=$PWD/../thirdparty/opencv/build/ \
    -DEIGEN_DIR=$PWD/../thirdparty/eigen/ \
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
$ git lfs pull
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DOpenCV_DIR=$PWD/../thirdparty/opencv/build/ \
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
    -DCMAKE_BUILD_TYPE=RELEASE \
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

Some of the benchmarks are ported from gcc-loops([link](https://github.com/llvm/llvm-test-suite/blob/main/SingleSource/UnitTests/Vectorizer/gcc-loops.cpp)) in LLVM test suit and linpackc([link](https://github.com/2000nickels/linpackc/blob/master/linpack.c))

*Note: Please replace the `/PATH/TO/*` with your local path and the `XXX` with specific target name (ex: gccloops,linpackc,matrix).*

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DVECTORIZATION_BENCHMARKS=ON \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/
$ ninja vectorization-XXX-benchmark
$ cd bin
$ ./vectorization-XXX-benchmark
```
## Gemmini Benchmark

Currently, we use the Spike simulator to run the Gemmini cases.
The cycle-accurate benchmark cases are working in the progress.
Before building the benchmark target, please see the following table and ensure you use the correct configuration.

| Cases | Hardware Configuration |
| -------------- | ------------- |
| Gemmini-ResNet-101  | defaultFpConfig ([link](./docs/GemminiConfig.md#using-default-float-point-configuration)) |

We assume you have already built all the components in the Gemmini README file. Now, let's build and run the cases. 

```
$ source /path/to/chipyard/env.sh
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/ \
    -DGEMMINI_BENCHMARKS=ON
$ ninja
$ cd bin
$ spike --extension=gemmini pk Gemmini-ResNet-101
```

## Operation Optimization Benchmark

Build and run MLIR operation optimization benchmark cases.

```
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DOP_OPTIMIZATION_BENCHMARKS=ON \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/
$ ninja <your target operation benchmark>

// Operation benchamrk supported include:
//   - conv2d-nchw-fchw-benchmark
//   - matmul-benchmark
```

Run TVM operation optimization benchmark cases.
- Install TVM ([steps](./thirdparty/README.md#tvm)).
- Enter to your TVM (virtual) environment.
- Configure TVM path and Python path.
- Navigate to your target operation directory (e.g. `buddy-benchmark/benchmarks/OpOptimization/MatMul/TVM`).
- (Optional) Configure the main file to specify the `target` or `size` of the benchmark.
- Run the main python file.

```
(tvm)$ export TVM_HOME=/path/to/tvm
(tvm)$ export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
(tvm)$ cd benchmarks/OpOptimization/<target operation>/TVM
(tvm)$ python main.py
```
