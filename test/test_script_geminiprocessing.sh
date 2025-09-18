#!/usr/bin/env bash

export BUDDY_MLIR_BUILD_DIR=/home/buddy-complier-workspace/buddy-mlir/build
export LLVM_MLIR_BUILD_DIR=/home/buddy-complier-workspace/buddy-mlir/llvm/build
export CHIPYARD_DIR=/home/buddy-complier-workspace/chipyard
export BUDDY_BENCHMARK_DIR=/home/buddy-complier-workspace/buddy-benchmark

cd "${CHIPYARD_DIR}"
git config --global --add safe.directory /home/buddy-complier-workspace/chipyard
git checkout 1.8.1

# Initialize and update the 'generators/gemmini' submodule and any submodules inside it.
git config --global --add safe.directory /home/buddy-complier-workspace/chipyard/generators/gemmini
git submodule update --init --recursive generators/gemmini

#############################################
# 1. Initialize Conda for the current shell
#############################################
conda init bash  # or "conda init" if you’re already in a bash shell

#############################################
# 2. Check if 'chipyard' environment exists
#############################################
if conda env list | grep -qE '^[^ ]*\s+chipyard\s'; then
    echo "[INFO] Found existing 'chipyard' environment. Activating it."
else
    echo "[INFO] 'chipyard' environment not found. Creating it..."
    # Example creation command - adjust packages as needed
    conda create -y -n chipyard python=3.10 \
        cmake ninja  \
        # plus any other dependencies needed...
fi

conda activate chipyard

#############################################
# 3. Source build-setup and env.sh
#############################################
# If your script uses conda-lock or has pinned requirements,
# you might need to call build-setup.sh so it *creates* the
# .conda-env environment. But be sure it doesn’t conflict
# with your newly created 'chipyard' environment.
source build-setup.sh esp-tools
source env.sh

#############################################
# 4. Proceed with your build
#############################################
cd "${BUDDY_BENCHMARK_DIR}"
rm -rf build
# Remove any existing build directory and create a fresh one.
mkdir -p build && cd build

RESULT_DIR="${BUDDY_BENCHMARK_DIR}/test_result/geminiprocessing"
mkdir -p "${RESULT_DIR}"

export C_PATH=$(which riscv64-unknown-linux-gnu-gcc)
export CXX_PATH=$(which riscv64-unknown-linux-gnu-g++)
export CLinker_PATH=$(which riscv64-unknown-linux-gnu-ld)

# Print Address here
echo "[Info] C_COMPILER_PATH = ${C_PATH}"
echo "[Info] CXX_COMPILER_PATH = ${CXX_PATH}"
echo "[Info] C_LINKER_PATH = ${CLinker_PATH}"
echo "[Info] BUDDY_MLIR_BUILD_DIR = ${BUDDY_MLIR_BUILD_DIR}"
echo "[Info] LLVM_MLIR_BUILD_DIR  = ${LLVM_MLIR_BUILD_DIR}"
echo "[Info] CHIPYARD_DIR = ${CHIPYARD_DIR}"
echo "[Info] BUDDY_BENCHMARK_DIR = ${BUDDY_BENCHMARK_DIR}"
echo "[Info] RESULT_DIR = ${RESULT_DIR}"

echo "[Info] Running CMake configuration..."
cmake -G Ninja .. \
  -DCMAKE_C_COMPILER=${C_PATH} \
  -DCMAKE_CXX_COMPILER=${CXX_PATH} \
  -DCMAKE_LINKER=${CLinker_PATH} \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
  -DGEMMINI_INCLUDE_DIR=${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests/include/ \
  -DGEMMINI_BENCHMARKS=ON \
  2>&1 | tee "${RESULT_DIR}/cmake_configure.log"

ninja 2>&1 | tee "${RESULT_DIR}/build.log"

# ```[1/21] Creating directories for 'project_googlebenchmark'
# [2/21] Building C object benchmarks/Gemmini/Ops/MatMulOp/CMakeFiles/ExoMatMul.dir/ExoMatmul.c.o
# FAILED: benchmarks/Gemmini/Ops/MatMulOp/CMakeFiles/ExoMatMul.dir/ExoMatmul.c.o 
# riscv64-unknown-linux-gnu-gcc  -I/home/buddy-complier-workspace/buddy-mlir/build/cmake/../../frontend/Interfaces -I/home/buddy-complier-workspace/buddy-mlir/build/cmake/../../thirdparty/include -I/home/buddy-complier-workspace/buddy-benchmark/benchmarks -I/home/buddy-complier-workspace/chipyard/generators/gemmini/software/gemmini-rocc-tests/include -I/home/buddy-complier-workspace/chipyard/generators/gemmini/software/gemmini-rocc-tests/include/.. -I/home/xychen/buddy-mlir/frontend/Interfaces -O3 -DNDEBUG -MD -MT benchmarks/Gemmini/Ops/MatMulOp/CMakeFiles/ExoMatMul.dir/ExoMatmul.c.o -MF benchmarks/Gemmini/Ops/MatMulOp/CMakeFiles/ExoMatMul.dir/ExoMatmul.c.o.d -o benchmarks/Gemmini/Ops/MatMulOp/CMakeFiles/ExoMatMul.dir/ExoMatmul.c.o -c /home/buddy-complier-workspace/buddy-benchmark/benchmarks/Gemmini/Ops/MatMulOp/ExoMatmul.c
# /home/buddy-complier-workspace/buddy-benchmark/benchmarks/Gemmini/Ops/MatMulOp/ExoMatmul.c: In function '_exo_matmul_4':
# /home/buddy-complier-workspace/buddy-benchmark/benchmarks/Gemmini/Ops/MatMulOp/ExoMatmul.c:28:47: error: macro "gemmini_extended_config_ex" requires 7 arguments, but only 6 given
#    28 |   gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
#       |                                               ^
# In file included from /home/buddy-complier-workspace/buddy-benchmark/benchmarks/Gemmini/Ops/MatMulOp/ExoMatmul.c:23:```

# cd bin
# ./vectorization-matrix-benchmark 2>&1 | tee "${RESULT_DIR}/run.log"

echo "[Info] CMake, build, and run logs are stored in ${RESULT_DIR}"
