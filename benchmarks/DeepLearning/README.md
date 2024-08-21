# Deep Learning Benchmark

## Model Level Benchmark
The table below lists the benchmark cases at the operation level.

| Name  | Build Target | Introduction |
| -------------- | ------------- | ------------- |
| MobileNet-V3 | `ninja dl-model-mobileNetV3-benchmark` | This benchmark compares multiple optimization strategies targeting the MobileNet-V3 model. |
| LeNet | `ninja dl-model-lenet-benchmark` | This benchmark compares multiple optimization strategies targeting the LeNet model. |

## Layer Level Benchmark
The table below lists the benchmark cases at the layer level.

| Name  | Build Target | Introduction |
| -------------- | ------------- | ------------- |
| FFN | `ninja dl-layer-ffn-benchmark` | This benchmark compares multiple optimization strategies targeting the FFN layer. |

## Operation Level Benchmark

The table below lists the benchmark cases at the operation level.

| Name  | Build Target | Introduction |
| -------------- | ------------- | ------------- |
| Linalg MatMul  | `ninja dl-op-linalg-matmul-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.matmul` operation. You can adjust the size of the benchmark by modifying the `M`, `N`, and `K` values in [this file](./Ops/MatMulOp/GoogleBenchmarkMain.cpp). |
| Linalg Conv2D NCHW FCHW | `ninja dl-op-linalg-conv2d-nchw-fchw-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.conv_2d_nchw_fchw` operation. You can adjust the size of the benchmark in [this file](./Ops/Conv2DNchwFchwOp/GoogleBenchmarkMain.cpp). |
| Linalg Conv2D NHWC HWCF | `ninja dl-op-linalg-conv2d-nhwc-hwcf-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.conv_2d_nhwc_hwcf` operation. You can adjust the size of the benchmark in [this file](./Ops/Conv2DNhwcHwcfOp/GoogleBenchmarkMain.cpp). |
| Linalg Depthwise Conv2D NHWC HWC | `ninja dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.depthwise_conv_2d_nhwc_hwc` operation. You can adjust the size of the benchmark in [this file](./Ops/DepthwiseConv2DNhwcHwcOp/GoogleBenchmarkMain.cpp). |
| Linalg Pooling NHWC Sum | `ninja dl-op-linalg-pooling-nhwc-sum-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.pooling_nhwc_sum` operation. You can adjust the size of the benchmark in [this file](./Ops/PoolingNhwcSumOp/GoogleBenchmarkMain.cpp). |
| Batch Matmul Benchmark | `ninja dl-op-linalg-batch-matmul-benchmark`  | This benchmark compares multiple optimization strategies targeting the `batch matmul` operation. You can adjust the size of the benchmark in [this file](./Ops/BatchMatMulOp/GoogleBenchmarkMain.cpp). |
| Arith Addf | `ninja dl-op-arith-addf-benchmark` | This benchmark evaluates optimization strategies for the `arith.addf` operation. The benchmark size can be adjusted in [this file](./Ops/ArithAddfOp/GoogleBenchmarkMain.cpp). |
| Arith Divf | `ninja dl-op-arith-divf-benchmark` | This benchmark evaluates optimization strategies for the `arith.divf` operation. The benchmark size can be adjusted in [this file](./Ops/ArithDivfOp/GoogleBenchmarkMain.cpp). |
| Arith Mulf | `ninja dl-op-arith-mulf-benchmark` | This benchmark evaluates optimization strategies for the `arith.mulf` operation. The benchmark size can be adjusted in [this file](./Ops/ArithMulfOp/GoogleBenchmarkMain.cpp). |
| Arith Negf | `ninja dl-op-arith-negf-benchmark` | This benchmark evaluates optimization strategies for the `arith.negf` operation. The benchmark size can be adjusted in [this file](./Ops/ArithNegfOp/GoogleBenchmarkMain.cpp). |
| Arith Subf | `ninja dl-op-arith-subf-benchmark` | This benchmark evaluates optimization strategies for the `arith.subf` operation. The benchmark size can be adjusted in [this file](./Ops/ArithSubfOp/GoogleBenchmarkMain.cpp). |
| Math Fpow | `ninja dl-op-math-fpow-benchmark` | This benchmark evaluates optimization strategies for the `math.fpow` operation. The benchmark size can be adjusted in [this file](./Ops/MathFpowOp/GoogleBenchmarkMain.cpp). |
| Math Rsqrt | `ninja dl-op-math-rsqrt-benchmark` | This benchmark evaluates optimization strategies for the `math.rsqrt` operation. The benchmark size can be adjusted in [this file](./Ops/MathRsqrtOp/GoogleBenchmarkMain.cpp). |
| Math Exp | `ninja dl-op-math-exp-benchmark` | This benchmark evaluates optimization strategies for the `math.exp` operation. The benchmark size can be adjusted in [this file](./Ops/MathExpOp/GoogleBenchmarkMain.cpp). |
| Reduce Addf | `ninja dl-op-reduce-addf-benchmark` | This benchmark evaluates optimization strategies for the `reduce.addf` operation. The benchmark size can be adjusted in [this file](./Ops/ReduceAddfOp/GoogleBenchmarkMain.cpp). |
| Reduce Maxf | `ninja dl-op-reduce-maxf-benchmark` | This benchmark evaluates optimization strategies for the `reduce.maxf` operation. The benchmark size can be adjusted in [this file](./Ops/ReduceMaxfOp/GoogleBenchmarkMain.cpp). |
| Softmax Exp Sum Div | `ninja dl-op-softmax-exp-sum-div-benchmark` | This benchmark evaluates optimization strategies for the `softmax.exp_sum_div` operation. The benchmark size can be adjusted in [this file](./Ops/SoftmaxExpSumDivOp/GoogleBenchmarkMain.cpp). |

### Enter Python virtual environment
We recommend you to use anaconda3 to create python virtual environment. You should install python packages as buddy-mlir/requirements.
```bash
$ conda activate <your virtual environment name>
$ cd buddy-benchmark
$ pip install -r requirements.txt
```

### Local Hardware Platform.

1. Set the `buddy-mlir` toolchain and PYTHONPATH environment variable:
Make sure that the PYTHONPATH variable includes the directory of LLVM/MLIR python bindings and the directory of Buddy MLIR python packages.

```bash
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

2. Build benchmark for local platform:

```bash
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}
$ ninja <target banchmark>
// For example: 
$ ninja dl-op-linalg-matmul-benchmark
```

3. Run the benchmark on your local platform:

```
// For example:
$ cd bin
$ ./dl-op-linalg-matmul-benchmark
```

### Cross Compile to Target Platform

**RISC-V Vector Extension**

Follow the relevant [documentation](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnviroment.md) to prepare the RVV environment.

1. Set variables for the toolchain:

```bash
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
$ export RISCV_GNU_TOOLCHAIN=${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
$ cd ../build-cross-rv
$ export BUDDY_MLIR_BUILD_CROSS_DIR=$PWD
```

2. Build the benchmark for the target platform:

```bash
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DCROSS_COMPILE_RVV=ON \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv \
    -DCMAKE_C_COMPILER=${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DBUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_CROSS_DIR}
$ ninja <target banchmark>
// For example: 
$ ninja dl-op-linalg-matmul-benchmark
```

3. Transfer the compiled benchmark to your target platform and run it.
