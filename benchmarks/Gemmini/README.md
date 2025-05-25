# Gemmini Benchmark
## How to Build

```bash
$ source /path/to/chipyard/env.sh
$ cd buddy-benchmark
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_BUILD_DIR=/PATH/TO/BUDDY-MLIR/BUILD/ \
    -DGEMMINI_INCLUDE_DIR=/PATH/TO/CHIPYARD/generators/gemmini/software/gemmini-rocc-tests/include/ \
    -DGEMMINI_BENCHMARKS=ON
$ ninja
```

The Gemmini Benchmark has two execution methods: one is using Spike, which can only verify functional correctness and can not provide accurate performance metrics; the other is using FireSim for execution.


## Spike
```bash
$ cd bin
$ spike --extension=gemmini pk dl-op-gemmini-matmul-benchmark
```

## Firesim

```bash
# Activate the FireSim conda environment
$ cd chipyard/sim/firesim 
$ source ./sourceme-manager.sh --skip-ssh-setup

# Copy the executable file into the overlay and recompile using Marshal
$ cd chipyard
$ cp ~/buddy-benchmark-gemmini/build/bin/dl-op-gemmini-matmul-benchmark ./generators/gemmini/software/overlay/root/BuddyGemmini/
$ ./sims/firesim/sw/firesim-software/marshal -v build ./generators/gemmini/software/gemmini-tests-interactive.json && ./sims/firesim/sw/firesim-software/marshal -v install ./generators/gemmini/software/gemmini-tests-interactive.json

#Start FireSim
$ firesim infrasetup
$ firesim runworkload
```
Then enter the FireSim shell and execute the executable file.