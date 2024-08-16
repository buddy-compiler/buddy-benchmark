# Buddy Compiler MobileNetV3 Benchmark

## MobileNetV3 Model Inference

1. Activate your python environment.
```bash
$ conda activate <your-python-env>
```

2. Build buddy-mlir

If you want to compile to RVV platform, Make sure you follow the relevant [documentation](../../README.md) to prepare the RVV environment and build buddy-mlir.

3. Set the `PYTHONPATH` environment variable.

Make sure you are in the buddy-mlir build directory.

```bash
$ export BUDDY_MLIR_BUILD_DIR=$PWD
$ export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
$ export PYTHONPATH=${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}
```

4. Build the MobileNetV3 benchmark

Make sure you are in the buddy-benchmark directory.

```bash
$ cd buddy-benchmark
$ cd build
$ ninja dl-model-mobileNetV3-benchmark
```

5. Run the MobileNetV3 benchmark

- Run on local platform.

Make sure you are in the buddy-benchmark build directory. Use the following command to run this benchmark.

```bash
$ ./bin/dl-model-mobileNetV3-benchmark
```

- Run on RISC-V platform. 

After transferring this ELF file to your RISC-V platform, use the following command to run this benchmark.
```bash
$ ./dl-model-mobileNetV3-benchmark
```