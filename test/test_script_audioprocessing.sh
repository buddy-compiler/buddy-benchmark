#!/usr/bin/env bash

export BUDDY_MLIR_BUILD_DIR=/home/buddy-complier-workspace/buddy-mlir/build
export LLVM_MLIR_BUILD_DIR=/home/buddy-complier-workspace/buddy-mlir/llvm/build
cd /home/buddy-complier-workspace/buddy-benchmark
mkdir -p build && cd build
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DAUDIO_PROCESSING_BENCHMARKS=ON \
    -DCMAKE_CXX_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang++ \
    -DKFR_DIR=/home/buddy-complier-workspace/buddy-benchmark/thirdparty/kfr \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR}
ninja dap-op-iir-benchmark
cd bin
./dap-op-iir-benchmark



cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DAUDIO_PROCESSING_BENCHMARKS=ON \
    -DCMAKE_CXX_COMPILER=${LLVM_MLIR_BUILD_DIR}/bin/clang++ \
    -DKFR_DIR=/home/buddy-complier-workspace/buddy-benchmark/thirdparty/kfr \
    -DBUDDY_MLIR_BUILD_DIR=${BUDDY_MLIR_BUILD_DIR} \
    -DPYTHON_BINARY_DIR="$(dirname "$(which python3)")"

ninja audio-plot
cd bin
./audio-plot ../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav ResultKFRIir.wav
# "
# root@4f445bb41579:/home/buddy-complier-workspace/buddy-benchmark/build/bin# ./audio-plot ../../benchmarks/AudioProcessing/Audios/NASA_Mars.wav ResultKFRIir.wav
# Plotting now...
# Traceback (most recent call last):
#   File "/home/buddy-complier-workspace/buddy-benchmark/utils/plots/python/plot.py", line 71, in <module>
#     compare_wave(args.file1, args.file2, part=args.part,
#   File "/home/buddy-complier-workspace/buddy-benchmark/utils/plots/python/plotools/compare.py", line 120, in compare_wave
#     after, time2 = get_time_domain(file2)
#   File "/home/buddy-complier-workspace/buddy-benchmark/utils/plots/python/plotools/compare.py", line 60, in get_time_domain
#     info, samples = get_info_and_samples(file)
#   File "/home/buddy-complier-workspace/buddy-benchmark/utils/plots/python/plotools/compare.py", line 38, in get_info_and_samples
#     with wave.open(file, 'rb') as audio:
#   File "/usr/lib/python3.10/wave.py", line 509, in open
#     return Wave_read(f)
#   File "/usr/lib/python3.10/wave.py", line 159, in __init__
#     f = builtins.open(f, 'rb')
# FileNotFoundError: [Errno 2] No such file or directory: 'ResultKFRIir.wav'
# "