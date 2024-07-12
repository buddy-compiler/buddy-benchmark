# Buddy Compiler MobileNetV3 Benchmark

## MobileNetV3 Model Inference

0. Activate your python environment.

1. Build buddy-mlir

```bash
$ cd buddy-mlir
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
$ ninja
$ ninja check-buddy
```

2. Set the `PYTHONPATH` environment variable.

Make sure you are in the build directory.

```bash
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

3. Set the `MOBILENETV3_EXAMPLE_PATH` environment variable.

```bash
$ cd buddy-benchmark
$ cd build
$ export MOBILENETV3_EXAMPLE_PATH=$PWD/../benchmarks/DeepLearning/MobileNet-V3/
```

4. Build and run the MobileNetV3 benchmark

```bash
$ ninja dl-model-mobileNetV3-benchmark
$ cd bin
$ ./dl-model-mobileNetV3-benchmark
```

