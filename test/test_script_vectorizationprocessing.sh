#!/usr/bin/env bash

################################################################################
# 1. Script Setup
################################################################################
set -e
BUDDY_MLIR_BUILD_DIR="/home/buddy-complier-workspace/buddy-mlir/build"
LLVM_MLIR_BUILD_DIR="/home/buddy-complier-workspace/buddy-mlir/llvm/build"

echo "[Info] BUDDY_MLIR_BUILD_DIR = ${BUDDY_MLIR_BUILD_DIR}"
echo "[Info] LLVM_MLIR_BUILD_DIR  = ${LLVM_MLIR_BUILD_DIR}"

RESULT_DIR="${PWD}/test_result/vectorization"
mkdir -p "${RESULT_DIR}"
LOG_FILE="${RESULT_DIR}/vectorization_result.log"
echo "Vectorization Benchmark - $(date)" > "${LOG_FILE}"

################################################################################
# 2. Build Benchmark
################################################################################
mkdir -p build && cd build
echo "[Info] Running CMake configuration..." | tee -a "${LOG_FILE}"
cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DVECTORIZATION_BENCHMARKS=ON \
  -DBUDDY_MLIR_BUILD_DIR="${BUDDY_MLIR_BUILD_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "[Info] Building vectorization-matrix-benchmark..." | tee -a "${LOG_FILE}"
ninja vectorization-matrix-benchmark 2>&1 | tee -a "${LOG_FILE}"

################################################################################
# 3. Run Benchmark
################################################################################
cd bin
echo "[Info] Running vectorization-matrix-benchmark..." | tee -a "${LOG_FILE}"
./vectorization-matrix-benchmark 2>&1 | tee -a "${LOG_FILE}"

echo "[Info] Benchmark completed. Log saved to ${LOG_FILE}"