#!/usr/bin/env bash

################################################################################
# 0. Script Setup
################################################################################
# We disable "exit on error" so that if one benchmark fails to build or run,
# we can continue with the rest.
set +e

################################################################################
# 1. (Optional) Activate Python/Conda Environment
################################################################################
# Uncomment or adjust if you use Anaconda/Miniconda:
# conda activate <YOUR-CONDA-ENV-NAME>


################################################################################
# 2. Build Each Benchmark (Continue Even If One Fails)
################################################################################
BENCHMARK_TARGETS=(
  # ------------------
  # Model-Level
  # ------------------
  "dl-model-tinyllama-benchmark"
  "dl-model-mobilenetv3-benchmark"
  "dl-model-lenet-benchmark"
  "dl-model-bert-benchmark"
  "dl-model-whisper-benchmark"
  "dl-model-resnet18-benchmark"

  # ------------------
  # Layer-Level
  # ------------------
  "dl-layer-ffn-benchmark"
  "dl-layer-selfattention-benchmark"
  "dl-layer-rmsnorm-benchmark"

  # ------------------
  # Operation-Level
  # ------------------
  "dl-op-linalg-matmul-benchmark"
  "dl-op-linalg-conv2d-nchw-fchw-benchmark"
  "dl-op-linalg-conv2d-nhwc-hwcf-benchmark"
  "dl-op-linalg-conv2d-nhwc-fhwc-benchmark"
  "dl-op-linalg-depthwise-conv-2d-nhwc-hwc-benchmark"
  "dl-op-linalg-pooling-nhwc-sum-benchmark"
  "dl-op-linalg-batch-matmul-benchmark"
  "dl-op-linalg-arithaddf-benchmark"
  "dl-op-linalg-arithdivf-benchmark"
  "dl-op-linalg-arithmulf-benchmark"
  "dl-op-linalg-arithnegf-benchmark"
  "dl-op-linalg-arithsubf-benchmark"
  "dl-op-linalg-mathfpow-benchmark"
  "dl-op-linalg-mathrsqrt-benchmark"
  "dl-op-linalg-mathexp-benchmark"
  "dl-op-linalg-reduceaddf-benchmark"
  "dl-op-linalg-reducemaxf-benchmark"
  "dl-op-linalg-softmax-exp-sum-div-benchmark"
  "dl-op-tosa-transpose-benchmark"
  "dl-op-matmul-transpose-b-benchmark"
)


################################################################################
# 3. Set Environment Variables for Buddy MLIR/LLVM
################################################################################
# Adjust these paths according to your local setup:
BUDDY_MLIR_DIR="/home/buddy-complier-workspace/buddy-mlir"  # The root directory of buddy-mlir
LLVM_BUILD_DIR="$BUDDY_MLIR_DIR/llvm/build"                 # The build dir for LLVM
BUDDY_BUILD_DIR="$BUDDY_MLIR_DIR/build"                     # The build dir for buddy-mlir

# Export environment variables:
export BUDDY_MLIR_BUILD_DIR="$BUDDY_BUILD_DIR"
export LLVM_MLIR_BUILD_DIR="$LLVM_BUILD_DIR"
export PYTHONPATH="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_BUILD_DIR}/python_packages:${PYTHONPATH}"
export BENCHMARK_PATH="${BUDDY_MLIR_DIR}/../buddy-benchmark"
echo "[Info] BUDDY_MLIR_BUILD_DIR = ${BUDDY_MLIR_BUILD_DIR}"
echo "[Info] LLVM_MLIR_BUILD_DIR  = ${LLVM_MLIR_BUILD_DIR}"
echo "[Info] PYTHONPATH           = ${PYTHONPATH}"

################################################################################
# 3. Prepare Build Folder and Run CMake
################################################################################
cd "${BUDDY_MLIR_DIR}/../buddy-benchmark" || exit 1
mkdir -p build
cd build || exit 1

echo "[Info] Running CMake configuration..."
cmake -G Ninja .. \
  -DDEEP_LEARNING_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUDDY_MLIR_BUILD_DIR="${BUDDY_MLIR_BUILD_DIR}" \
  -DCMAKE_CXX_COMPILER="${LLVM_MLIR_BUILD_DIR}/bin/clang++" \
  -DCMAKE_C_COMPILER="${LLVM_MLIR_BUILD_DIR}/bin/clang" \
  -DCMAKE_CXX_FLAGS="-march=native" \
  -DCMAKE_C_FLAGS="-march=native"


################################################################################
# 4. Prepare Build Folder and Run CMake
################################################################################

mkdir -p $BENCHMARK_PATH/test_result
mkdir -p $BENCHMARK_PATH/test_result/deeplearning
BUILD_LOG="${BENCHMARK_PATH}/test_result/deeplearning/build_results_summary.log"
> "${BUILD_LOG}"  # Clear/create the file

echo "[Info] Building all benchmarks with Ninja..."
for target in "${BENCHMARK_TARGETS[@]}"; do
  echo "==> ninja ${target}"
  if ninja "${target}"; then
    echo "[Success] Build of '${target}'" | tee -a "${BUILD_LOG}"
  else
    echo "[Failed]  Build of '${target}'" | tee -a "${BUILD_LOG}"
  fi
done

################################################################################
################################################################################
# 5. Run Each Benchmark & Redirect Output (Continue Even If One Fails)
################################################################################
cd bin || exit 1

RUN_LOG="${BENCHMARK_PATH}/test_result/deeplearning/run_results_summary.log"
> "${RUN_LOG}"        # clear / create the file

echo "[Info] Running all benchmarks in ./bin..."
for target in "${BENCHMARK_TARGETS[@]}"; do
  if [[ -f "${target}" ]]; then
    echo "==> Running ${target}"

    # ---- NEW: dump a machine-readable report next to the plain log -----------
    json_out="${BENCHMARK_PATH}/test_result/deeplearning/${target}.json"

    if "./${target}" \
          --benchmark_out="${json_out}" \
          --benchmark_out_format=json \
          >  "${BENCHMARK_PATH}/test_result/deeplearning/${target}.log" 2>&1
    then
      echo "[Success] Run of '${target}'"  | tee -a "${RUN_LOG}"
      echo "         ↳ stdout/stderr → ${target}.log"  | tee -a "${RUN_LOG}"
      echo "         ↳ gbench JSON   → ${target}.json" | tee -a "${RUN_LOG}"
    else
      echo "[Failed]  Run of '${target}'"  | tee -a "${RUN_LOG}"
      echo "         ↳ stdout/stderr → ${target}.log (may contain errors)" | tee -a "${RUN_LOG}"
    fi
    # -------------------------------------------------------------------------
  else
    echo "[Missing] Executable not found for '${target}'" | tee -a "${RUN_LOG}"
  fi
done


################################################################################
# 6. Set Environment Variables for Buddy MLIR/LLVM for cross-compile
################################################################################
# Adjust these paths according to your local setup:
BUDDY_MLIR_DIR="/home/buddy-complier-workspace/buddy-mlir"  # The root directory of buddy-mlir
LLVM_BUILD_DIR="$BUDDY_MLIR_DIR/llvm/build"                 # The build dir for LLVM
BUDDY_BUILD_DIR="$BUDDY_MLIR_DIR/build"                     # The build dir for buddy-mlir

# Export environment variables:
export BUDDY_MLIR_BUILD_DIR="$BUDDY_BUILD_DIR"
export LLVM_MLIR_BUILD_DIR="$LLVM_BUILD_DIR"
export PYTHONPATH="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${BUDDY_BUILD_DIR}/python_packages:${PYTHONPATH}"
export BUDDY_MLIR_BUILD_CROSS_DIR=${BUDDY_MLIR_BUILD_DIR}/../build
export RISCV_GNU_TOOLCHAIN=${BUDDY_MLIR_BUILD_DIR}/../thirdparty/riscv-gnu-toolchain
export RISCV_OMP_SHARED=${LLVM_MLIR_BUILD_DIR}/../build/lib/libomp.so
export BENCHMARK_PATH="${BUDDY_MLIR_DIR}/../buddy-benchmark"

echo "[Info] BUDDY_MLIR_BUILD_DIR = ${BUDDY_MLIR_BUILD_DIR}"
echo "[Info] LLVM_MLIR_BUILD_DIR  = ${LLVM_MLIR_BUILD_DIR}"
echo "[Info] PYTHONPATH           = ${PYTHONPATH}"

################################################################################
# 7. Prepare Build Folder and Run CMake
################################################################################
cd "${BUDDY_MLIR_DIR}/../buddy-benchmark" || exit 1
mkdir -p build
cd build || exit 1

echo "[Info] Running CMake configuration..."
cmake -G Ninja .. \
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

################################################################################
# 8. Prepare Build Folder and Run CMake for cross-compile
################################################################################

mkdir -p $BENCHMARK_PATH/test_result
BUILD_LOG="${BENCHMARK_PATH}/test_result/deeplearning/build_results_crosscompile_summary.log"
> "${BUILD_LOG}"  # Clear/create the file

echo "[Info] Building all benchmarks with Ninja..."
for target in "${BENCHMARK_TARGETS[@]}"; do
  echo "==> ninja ${target}"
  if ninja "${target}"; then
    echo "[Success] Build of '${target}'" | tee -a "${BUILD_LOG}"
  else
    echo "[Failed]  Build of '${target}'" | tee -a "${BUILD_LOG}"
  fi
done


echo
echo "[Info] All build/run steps completed (script did not stop on failures)."
echo "[Info] Build summary: ${BUILD_LOG}"
echo "[Info] Run summary:   ${RUN_LOG}"


cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)