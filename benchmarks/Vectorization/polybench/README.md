# Polybench Benchmark

This folder contains the port of the Polybench benchmark suite.

## Build the Benchmark

### Local Hardware Platform

1. Set the `buddy-mlir` toolchain environment variables:

```bash
cd buddy-mlir/build
export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build
```

2. Build benchmark for local platform:

```bash
cd buddy-benchmark
mkdir build && cd build
cmake -G Ninja .. \
    -DVECTORIZATION_BENCHMARKS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DCMAKE_CXX_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang++ \
    -DCMAKE_C_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang \
    -DCMAKE_CXX_FLAGS=-march=native \
    -DCMAKE_C_FLAGS=-march=native
ninja vectorization-polybench-benchmark
```

1. Run the benchmark, check the [Running the Benchmark](#running-the-benchmark)
   section for more details.

### Cross Compile to Target Platform

**RISC-V Vector Extension**

Follow the relevant
[documentation](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnviroment.md)
to prepare the RVV environment.

1. Set variables for the toolchain:

```bash
cd buddy-mlir/build
export BUDDY_MLIR_BUILD_DIR=$PWD
export LLVM_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}/../llvm/build/
export BUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_DIR}/../build-cross-rv
export RISCV_GNU_TOOLCHAIN=${BUDDY_MLIR_BUILD_DIR}/thirdparty/riscv-gnu-toolchain
```

2. Build the benchmark for the target platform:

```bash
cd buddy-benchmark
mkdir build && cd build
cmake -G Ninja .. \
    -DVECTORIZATION_BENCHMARKS=ON \
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
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DBUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_CROSS_DIR} \
    -DBUDDY_MLIR_CROSS_LIB_DIR=${BUDDY_MLIR_BUILD_CROSS_DIR}/lib
ninja vectorization-polybench-benchmark
```

3. Transfer the compiled binary to the target platform and run it.

## Running the Benchmark

The binary `vectorization-polybench-benchmark` inherits the command-line options
from google benchmark including the benchmark filtering feature (see
[here](https://github.com/google/benchmark/blob/main/docs/user_guide.md#running-a-subset-of-benchmarks)
for more details). The benchmark cases are organized as
`MLIRPolybenchCase/methods/<dataset_size_id>`, so the following command can be used
to run the benchmark for all cases on `small` and `medium` dataset sizes
(indices 1 and 2).

```bash
./bin/vectorization-polybench-benchmark --benchmark_filter=".*/.*/[12]"
```

And to run the benchmark for a specific case (e.g., `2mm`) on a specific dataset
sizes (e.g., `mini`, `small` and `medium`) with all methods, use the following
command:

```bash
./bin/vectorization-polybench-benchmark --benchmark_filter="MLIRPolybench2mm/.*/[012]"
```

Also, to verify the correctness of different methods, use
`--verification-dataset-size` option. The command below runs all benchmark cases
on `small` and `medium` dataset sizes, and verifies the results using the `mini`
dataset size for optimized methods.

```bash
./bin/vectorization-polybench-benchmark --benchmark_filter=".*/.*/[12]" --verification-dataset-size=mini
```

Additionally, `--generate-output` runs the benchmark with the specified dataset 
size and output the result to stdout. The format of output data is the same as
the original Polybench suite. So this output can be used to validate the 
correctness of this port.

```bash
./bin/vectorization-polybench-benchmark --generate-output=mini
```

Note that the output can be very large, so it is recommended to redirect the
output to a file and compare it with the original Polybench output later.

## Verifying the Benchmark Results

`polybench_mlir_gen.py` is provided to generate the output of the original
Polybench suite for the specified dataset size. Run the following command to
generate the output for the mini dataset size:

```bash
python benchmarks/Vectorization/polybench/polybench_mlir_gen.py \
    --output-dir output \
    --polygeist-build-dir ${POLYGEIST_BUILD_DIR} \
    --polybench-dir ${POLYBENCH_SRC_DIR} \
    --generate-mlir \
    --generate-binary \
    --binary-compiler=cgeist \
    --generate-std-output \
    --std-output-file "polybench-mini.txt" \
    --std-output-dataset-size mini
```

This will generate `output/polybench-mini.txt` which contains the output of the
original Polybench suite for the mini dataset size. Then, run the following
command to generate the output of the ported Polybench suite:

```bash
<path_to_build>/bin/vectorization-polybench-benchmark --generate-output=mini > output/ported-mini.txt
```

The `output/polybench-mini.txt` and `ported-mini.txt` can be compared using diff
to verify the correctness of the port.

Note that the python script generates the output with `-O0` and it is recommended
to use cgeist as the binary compiler to generate the output (an alternative is
to use `--binary-compiler=clang`, which might lead to floating-point differences).

## About this Benchmark

Most of the Polybench cases uses parametric loop bounds and sizes, which makes
it easy to run the benchmark with different dataset sizes. But there are some
that uses compile-time constant sizes (including fix-sized local arrays). The
MLIR source code here are generated by
[Polygeist](https://github.com/llvm/Polygeist) and modified manually to run on
different dataset sizes dynamically.
