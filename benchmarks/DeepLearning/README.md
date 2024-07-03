# Deep Learning Benchmark

## Dependencies

Ensure the following dependencies are ready:
<!-- TODO: Remove OpenCV dependency -->
### OpenCV

```
$ cd buddy-benchmark
$ cd thirdparty/opencv
$ mkdir build && cd build
$ cmake -G Ninja ..
$ ninja
```

## Operation Level Benchmark

The table below lists the benchmark cases at the operation level.

| Name  | Build Target | Introduction |
| -------------- | ------------- | ------------- |
| Linalg MatMul  | `ninja dl-op-linalg-matmul-benchmark`  | This benchmark compares multiple optimization strategies targeting the `linalg.matmul` operation. You can adjust the size of the benchmark by modifying the `M`, `N`, and `K` values in [this file](./Ops/MatMulOp/GoogleBenchmarkMain.cpp). |

### Local Hardware Platform.

1. Set the `buddy-mlir` toolchain:

```
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
```

2. Build benchmark for local platform:

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DOpenCV_DIR=$PWD/../thirdparty/opencv/build/ \
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

```
$ cd buddy-mlir/build
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export RISCV_GNU_TOOLCHAIN=${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
```

2. Build the benchmark for the target platform:

```
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DDEEP_LEARNING_BENCHMARKS=ON \
    -DOpenCV_DIR=$PWD/../thirdparty/opencv/build/ \
    -DCROSS_COMPILE_RVV=ON \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv \
    -DCMAKE_C_COMPILER=${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++ \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}
$ ninja <target banchmark>
// For example: 
$ ninja dl-op-linalg-matmul-benchmark
```

3. Transfer the compiled benchmark to your target platform and run it.
