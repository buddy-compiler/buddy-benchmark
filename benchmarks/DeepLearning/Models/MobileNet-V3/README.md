# Buddy Compiler MobileNetV3 Benchmark

## MobileNetV3 Model Inference

0. Activate your python environment.

1. Build buddy-mlir

Make sure you follow the relevant [documentation](../../README.md) to prepare the RVV environment and build buddy-mlir.

2. Set the `PYTHONPATH` environment variable.

Make sure you are in the buddy-mlir build directory.

```bash
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

4. Build and run the MobileNetV3 benchmark

```bash
$ cd buddy-benchmark/build
$ ninja dl-model-mobileNetV3-benchmark
$ ./bin/dl-model-mobileNetV3-benchmark
```

