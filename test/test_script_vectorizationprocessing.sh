#!/usr/bin/env bash

# apt update
# apt install -y libc6-riscv64-cross
# apt install -y \
#      libc6-riscv64-cross       \
#      libstdc++6-riscv64-cross  \
#      libgcc-s1-riscv64-cross 
################################################################################
# 1. Script Setup
################################################################################
set -e
BUDDY_MLIR_BUILD_DIR="/home/buddy-complier-workspace/buddy-mlir/build"
LLVM_MLIR_BUILD_DIR="/home/buddy-complier-workspace/buddy-mlir/llvm/build"
# Export environment variables:
PYTHONPATH="${LLVM_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_MLIR_BUILD_DIR}/python_packages:${PYTHONPATH}"
BUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_DIR}/../build
RISCV_GNU_TOOLCHAIN=${BUDDY_MLIR_BUILD_DIR}/../thirdparty/riscv-gnu-toolchain
RISCV_OMP_SHARED=${LLVM_MLIR_BUILD_DIR}/../build/lib/libomp.so
BENCHMARK_PATH="${BUDDY_MLIR_DIR}/../buddy-benchmark"

echo "[Info] BUDDY_MLIR_BUILD_DIR = ${BUDDY_MLIR_BUILD_DIR}"
echo "[Info] LLVM_MLIR_BUILD_DIR  = ${LLVM_MLIR_BUILD_DIR}"

RESULT_DIR="${PWD}/test_result/vectorization"
mkdir -p "${RESULT_DIR}"
LOG_FILE="${RESULT_DIR}/vectorization_result.log"
echo "Vectorization Benchmark - $(date)" > "${LOG_FILE}"

################################################################################
# 2. Build Benchmark
################################################################################
cd /home/buddy-complier-workspace/buddy-benchmark
echo "[Info] Starting vectorization-matrix-benchmark build..." | tee -a "${LOG_FILE}"
rm -rf build
mkdir -p build && cd build
echo "[Info] Running CMake configuration..." | tee -a "${LOG_FILE}"
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DVECTORIZATION_BENCHMARKS=ON \
  -DBUDDY_MLIR_BUILD_DIR="${BUDDY_MLIR_BUILD_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "[Info] Building vectorization-matrix-benchmark..." | tee -a "${LOG_FILE}"
ninja vectorization-matrix-benchmark 2>&1 | tee -a "${LOG_FILE}"

export QEMU_LD_PREFIX=/usr/riscv64-linux-gnu
################################################################################
# 3. Run Benchmark
################################################################################
cd bin
echo "[Info] Running vectorization-matrix-benchmark..." | tee -a "${LOG_FILE}"
./vectorization-matrix-benchmark 2>&1 | tee -a "${LOG_FILE}"

echo "[Info] Benchmark completed. Log saved to ${LOG_FILE}"