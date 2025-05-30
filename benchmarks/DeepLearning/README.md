# Deep Learning Benchmark

## Benchmark Lists
### Model Level Benchmark
The table below lists the benchmark cases at the operation level.

| Name  | Build Target | Introduction |
| -------------- | ------------- | ------------- |
| TinyLlama-1.1B | `ninja dl-model-tinyllama-benchmark` | This benchmark compares multiple optimization strategies targeting the TinyLlama model. |
| MobileNet-V3 | `ninja dl-model-mobilenetv3-benchmark` | This benchmark compares multiple optimization strategies targeting the MobileNet-V3 model. |
| LeNet | `ninja dl-model-lenet-benchmark` | This benchmark compares multiple optimization strategies targeting the LeNet model. |
| BERT | `ninja dl-model-bert-benchmark` | This benchmark compares multiple optimization strategies targeting the BERT model. |
| Whisper | `ninja dl-model-whisper-benchmark` | This benchmark compares multiple optimization strategies targeting the Whisper model. |
| ResNet-18 | `ninja dl-model-resnet18-benchmark` | This benchmark compares multiple optimization strategies targeting the ResNet-18 model. |

### Layer Level Benchmark
The table below lists the benchmark cases at the layer level.

| Name  | Build Target | Introduction |
| -------------- | ------------- | ------------- |
| FFN | `ninja dl-layer-ffn-benchmark` | This benchmark compares multiple optimization strategies targeting the FFN layer. |
| Self Attention | `ninja dl-layer-selfattention-benchmark` | This benchmark compares multiple optimization strategies targeting the self attention layer. |
| RMSNorm | `ninja dl-layer-rmsnorm-benchmark` | This benchmark compares multiple optimization strategies targeting the RMSNorm layer. |

### Operation Level Benchmark

The table below lists the benchmark cases at the operation level.

| Name  | Build Target | Introduction |
| -------------- | ------------- | ------------- |
| Linalg MatMul  | `ninja dl-op-linalg-matmul-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.matmul` operation. You can adjust the size of the benchmark by modifying the `M`, `N`, and `K` values in [this file](./Ops/MatMulOp/GoogleBenchmarkMain.cpp). |
| Linalg MatMul for int32 data type by RVV optimization | `ninja dl-op-linalg-matmul-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.matmul` operation for int32 data type by RVV optimization. You can adjust the size of the benchmark by modifying the `M`, `N`, and `K` values in [this file](./Ops/MatMulInt32Op/Main.cpp). |
| Linalg Conv2D NCHW FCHW | `ninja dl-op-linalg-conv2d-nchw-fchw-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.conv_2d_nchw_fchw` operation. You can adjust the size of the benchmark in [this file](./Ops/Conv2DNchwFchwOp/GoogleBenchmarkMain.cpp). |
| Linalg Conv2D NHWC HWCF | `ninja dl-op-linalg-conv2d-nhwc-hwcf-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.conv_2d_nhwc_hwcf` operation. You can adjust the size of the benchmark in [this file](./Ops/Conv2DNhwcHwcfOp/GoogleBenchmarkMain.cpp). |
| Linalg Conv2D NHWC FHWC | `ninja dl-op-linalg-conv2d-nhwc-fhwc-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.conv_2d_nhwc_fhwc` operation. You can adjust the size of the benchmark in [this file](./Ops/Conv2DNhwcFhwcOp/Main.cpp). |
| Linalg Conv2D NHWC FHWC for int32 data type by RVV optimization| `ninja dl-op-linalg-conv2d-nhwc-fhwc-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.conv_2d_nhwc_fhwc` operation for int32 data type by RVV optimization. You can adjust the size of the benchmark in [this file](./Ops/Conv2DNhwcFhwcInt32Op/Main.cpp). |
| Linalg Depthwise Conv2D NHWC HWC | `ninja dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.depthwise_conv_2d_nhwc_hwc` operation. You can adjust the size of the benchmark in [this file](./Ops/DepthwiseConv2DNhwcHwcOp/Main.cpp). |
| Linalg Pooling NHWC Sum | `ninja dl-op-linalg-pooling-nhwc-sum-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.pooling_nhwc_sum` operation. You can adjust the size of the benchmark in [this file](./Ops/PoolingNhwcSumOp/GoogleBenchmarkMain.cpp). |
| Linalg Batch Matmul Benchmark | `ninja dl-op-linalg-batch-matmul-benchmark`  | This benchmark compares multiple optimization strategies targeting the `batch matmul` operation. You can adjust the size of the benchmark in [this file](./Ops/BatchMatMulOp/Main.cpp). |
| Linalg Batch Matmul Benchmark for int32 data type by RVV optimization  | `ninja dl-op-linalg-batch-matmul-benchmark`  | This benchmark compares multiple optimization strategies targeting the `batch matmul` operation for int32 data type by RVV optimization. You can adjust the size of the benchmark in [this file](./Ops/BatchMatMulInt32Op/Main.cpp). |
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
| TOSA Transpose | `ninja dl-op-tosa-transpose-benchmark` | This benchmark evaluates optimization strategies for the `tosa.transpose` operation. The benchmark size can be adjusted in [this file](./Ops/TransposeOp/Main.cpp). |
| MatMul Transpose B | `ninja dl-op-matmul-transpose-b-benchmark` | This benchmark evaluates optimization strategies for the `linalg.matmul_transpose_b` operation. The benchmark size can be adjusted in [main file](./Ops/MatMulTransposeBOp/Main.cpp) and [MLIR file](./Ops/MatMulTransposeBOp/MatMulTransposeB.mlir). |
| Linalg Batch MatMul Transpose B | `ninja dl-op-linalg-batch-matmul-transpose-b-benchmark` | This benchmark evaluates optimization strategies for the `linalg.batch_matmul_transpose_b` operation. The benchmark size can be adjusted in [main file](./Ops/BatchMatMulTransposeBOp/Main.cpp) and [MLIR file](./Ops/BatchMatMulTransposeBOp/BatchMatMulTransposeB.mlir). |

## How to Build

### Enter Python virtual environment
We recommend you to use anaconda3 to create python virtual environment. You should install python packages as buddy-mlir/requirements.
```bash
$ conda activate <your virtual environment name>
$ cd buddy-benchmark
$ pip install -r requirements.txt
```

### Build on Local Hardware Platform

1. Set the `buddy-mlir` toolchain and PYTHONPATH environment variable:
Make sure that the PYTHONPATH variable includes the directory of LLVM/MLIR python bindings and the directory of Buddy MLIR python packages.

```bash
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}/../llvm/build/
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

2. Build benchmark for local platform:

```bash
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DCMAKE_CXX_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang++ \
    -DCMAKE_C_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang \
    -DCMAKE_CXX_FLAGS=-march=native \
    -DCMAKE_C_FLAGS=-march=native
$ ninja <target benchmark>
// For example: 
$ ninja dl-op-linalg-matmul-benchmark
```

3. Run the benchmark on your local platform:

```bash
// For example:
$ cd bin
$ ./dl-op-linalg-matmul-benchmark
```

### Cross Compile to Target Platform

**RISC-V Vector Extension**

Follow the [Environment Setup Guide for MLIR and RVV Testing and Experiments](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md) to prepare the RVV environment. Furthermore, To enable the openmp feature on RISC-V, you also need to refer to [Prepare RISC-V OpenMP ToolChain](../../docs/PrepareRVOpenMP.md).

1. Set variables for the toolchain:

```bash
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}/../llvm/build/
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
$ export BUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_DIR}/../build-cross-rv
$ export RISCV_GNU_TOOLCHAIN=${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
$ export RISCV_OMP_SHARED=${LLVM_MLIR_BUILD_DIR}/../build-omp-shared-rv/libomp.so

```

2. Build the benchmark for the target platform:

```bash
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCROSS_COMPILE_RVV=ON \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv \
    -DCMAKE_C_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang \
    -DRISCV_GNU_TOOLCHAIN=${RISCV_GNU_TOOLCHAIN} \
    -DCMAKE_CXX_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="-march=rv64gcv --target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot --gcc-toolchain=${RISCV_GNU_TOOLCHAIN} -fPIC" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv --target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot --gcc-toolchain=${RISCV_GNU_TOOLCHAIN} -fPIC" \
    -DRISCV_OMP_SHARED=${RISCV_OMP_SHARED} \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DBUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_CROSS_DIR} \
    -DBUDDY_MLIR_CROSS_LIB_DIR=${BUDDY_MLIR_BUILD_CROSS_DIR}/lib

$ ninja <target benchmark>
// For example: 
$ ninja dl-op-linalg-matmul-benchmark
```

3. Transfer the compiled benchmark to your target platform and run it.
