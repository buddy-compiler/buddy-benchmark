# IntrinsicBenchmark

## Build

``` bash
git submodule update --init --recursive
cmake -G Ninja -B build-native-rv . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain/riscv.cmake -DBUILD_VERSION=native
cmake -G Ninja -B build-scalar-rv . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain/riscv.cmake -DBUILD_VERSION=scalar

cmake -G Ninja -B build-native-aarch64 . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain/aarch64.cmake -DBUILD_VERSION=native
cmake -G Ninja -B build-scalar-aarch64 . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain/aarch64.cmake -DBUILD_VERSION=scalar

cmake -G Ninja -B build-native-sse . -DCMAKE_BUILD_TYPE=Release -DBUILD_VERSION=native -DTARGET_ARCH=sse
cmake -G Ninja -B build-native-avx2 . -DCMAKE_BUILD_TYPE=Release -DBUILD_VERSION=native -DTARGET_ARCH=avx2
cmake -G Ninja -B build-native-avx512 . -DCMAKE_BUILD_TYPE=Release -DBUILD_VERSION=native -DTARGET_ARCH=avx512
cmake -G Ninja -B build-scalar-x86 . -DCMAKE_BUILD_TYPE=Release -DBUILD_VERSION=scalar

cmake -G Ninja -B build-custom . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain/riscv.cmake -DBUILD_VERSION=custom -DCMAKE_CXX_FLAGS="-march=rv64gcv"
```

Note: Assume that there are GNU cross-compilation toolchains for RISC-V and ARM platforms in `/opt/riscv/bin` and `/opt/aarch64/bin`. Otherwise the bin directory should be specified via the CMake variable `TOOLCHAIN_COMPILER_LOCATION_HINT`

## Run

Assume that the host machine can access the target device via ssh, then

```bash
# Test on RSIC-V device
python3 ./run.py --host <DEVICE_IP> --user <DEVICE_USER_NAME> --program ./build-native-rv/intrinsicPerf --output ./output/native-rv.json
python3 ./run.py --host <DEVICE_IP> --user <DEVICE_USER_NAME> --program ./build-scalar-rv/intrinsicPerf --output ./output/scalar-rv.json
python3 ./run.py --host <DEVICE_IP> --user <DEVICE_USER_NAME> --program ./build-custom/intrinsicPerf --output ./output/custom.json
python3 ./report.py output/scalar-rv.json output/native-rv.json ./output/custom.json --output result/rv.csv

# Test on ARM device
python3 ./run.py --host <DEVICE_IP> --user <DEVICE_USER_NAME> --program ./build-native-aarch64/intrinsicPerf --output ./output/native-aarch64.json
python3 ./run.py --host <DEVICE_IP> --user <DEVICE_USER_NAME> --program ./build-scalar-aarch64/intrinsicPerf --output ./output/scalar-aarch64.json
python3 ./report.py output/scalar-aarch64.json output/native-aarch64.json --output result/aarch64.csv

```
