#!/usr/bin/env bash

# NEW: Create results directory and update log file path
RESULT_DIR="${PWD}/test_result/imageprocessing"
mkdir -p "$RESULT_DIR"
LOG="${RESULT_DIR}/image-processing-result.log"
echo "Benchmark results - $(date)" > "$LOG"

# Function to check CPU flag support
supports() {
    local flag=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    if grep -qi "$flag" /proc/cpuinfo; then
         return 0
    else
         return 1
    fi
}

features=("SSE" "AVX2" "AVX512" "NEON")
images=("../benchmarks/ImageProcessing/Images/YuTu.png")
kernels=("prewittKernelAlign" "sobel3x3KernelAlign" "sobel5x5KernelAlign" "sobel7x7KernelAlign" "sobel9x9KernelAlign" "laplacianKernelAlign" "logKernelAlign")
kernelmorphs=("random3x3KernelAlignInt")
boundaries=("CONSTANT_PADDING" "REPLICATE_PADDING")

for feature in "${features[@]}"; do
   echo "Testing $feature support" | tee -a "$LOG"
   if supports "$feature"; then
      echo "$feature is supported." | tee -a "$LOG"
      mkdir -p build_${feature} && cd build_${feature}
      cmake -G Ninja .. \
          -DCMAKE_BUILD_TYPE=RELEASE \
          -DIMAGE_PROCESSING_BENCHMARKS=ON \
          -DOpenCV_DIR=$PWD/../thirdparty/opencv/build/ \
          -DEIGEN_DIR=$PWD/../thirdparty/eigen/ \
          -DBUDDY_OPT_ATTR=$(echo "$feature" | tr '[:upper:]' '[:lower:]') \
          -DBUDDY_MLIR_BUILD_DIR=/home/buddy-complier-workspace/buddy-mlir/build
      ninja image-processing-benchmark
      echo "Running image-processing-benchmark for $feature" | tee -a "$LOG"
      # ---------------------------------------------------------------------------
      # inside the big loop – ONLY this section is changed
      # ---------------------------------------------------------------------------
      for img in "${images[@]}"; do
        img_slug=$(basename "$img" .png)             # YuTu → YuTu
        for kern in "${kernels[@]}"; do
          for morph in "${kernelmorphs[@]}"; do
            for boundary in "${boundaries[@]}"; do
              echo "Running: $img $kern $morph $boundary" | tee -a "$LOG"

              # ---- NEW: build a unique JSON filename ---------------------------------
              slug="$(echo "${feature}_${img_slug}_${kern}_${morph}_${boundary}" \
                    | tr ' /' '__')"
              json_out="${RESULT_DIR}/${slug}.json"
              log_out="${RESULT_DIR}/${slug}.log"
              # -----------------------------------------------------------------------

              ./bin/image-processing-benchmark  \
                  "$img" "$kern" "$morph" "$boundary" \
                  --benchmark_out="$json_out" \
                  --benchmark_out_format=json \
                  >  "$log_out" 2>&1
              echo "[Success] …"        | tee -a "$LOG"
            done
          done
        done
      done
      cd ..
   else
       echo "CPU does not support $feature." | tee -a "$LOG"
   fi
done

# NEW: Clean up build directories
for feature in "${features[@]}"; do
    rm -rf "build_${feature}"
done