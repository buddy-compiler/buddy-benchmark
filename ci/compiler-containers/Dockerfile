FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates gnupg ninja-build gawk \
           build-essential cmake make python3 python3-dev python3-pip zlib1g wget subversion unzip git \
           autoconf automake autotools-dev curl libmpc-dev libmpfr-dev libgmp-dev bison \
           libfdt-dev libpixman-1-dev libcapstone-dev\
           flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev libglib2.0-dev libslirp-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install numpy pybind11 PyYAML

    
# 0. Prepare buddy-mlir and Submodules
# RUN git clone https://github.com/buddy-compiler/buddy-mlir.git && \
#     cd buddy-mlir && \
#     git submodule update --init
COPY buddy-mlir /opt/buddy-mlir
ENV BUILD_ROOT=/opt

# 1. Build Local LLVM/MLIR
RUN mkdir ${BUILD_ROOT}/buddy-mlir/llvm/build
WORKDIR ${BUILD_ROOT}/buddy-mlir/llvm/build
RUN cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
     -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
     -DCMAKE_BUILD_TYPE=RELEASE \
     -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
     -DPython3_EXECUTABLE=$(which python3)
RUN ninja check-clang check-mlir omp
ENV BUILD_LOCAL_LLVM_DIR=${BUILD_ROOT}/buddy-mlir/llvm/build

# 2. Build Local buddy-mlir
RUN mkdir ${BUILD_ROOT}/buddy-mlir/build
WORKDIR ${BUILD_ROOT}/buddy-mlir/build
RUN cmake -G Ninja .. \
     -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
     -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DCMAKE_BUILD_TYPE=RELEASE \
     -DBUDDY_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN=ON \
     -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
     -DPython3_EXECUTABLE=$(which python3)       
RUN ninja -j1
RUN ninja check-buddy
ENV BUILD_RISCV_GNU_TOOLCHAIN_DIR=${BUILD_ROOT}/buddy-mlir/build/thirdparty/riscv-gnu-toolchain/
ENV RISCV_GNU_TOOLCHAIN_SYSROOT_DIR=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}/sysroot/


# 3. Build Cross-Compiled Clang
RUN mkdir ${BUILD_ROOT}/buddy-mlir/llvm/build-cross-clang-rv
WORKDIR ${BUILD_ROOT}/buddy-mlir/llvm/build-cross-clang-rv
RUN cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS="clang" \
     -DLLVM_TARGETS_TO_BUILD="RISCV" \
     -DCMAKE_SYSTEM_NAME=Linux \
     -DCMAKE_C_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang \
     -DCMAKE_CXX_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang++ \
     -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
     -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
     -DLLVM_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/llvm-tblgen \
     -DCLANG_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/clang-tblgen \
     -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
     -DLLVM_TARGET_ARCH=RISCV64 \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ZSTD=Off
RUN ninja clang lli

# 4. Build Cross-Compiled MLIR
RUN mkdir llvm/build-cross-mlir-rv
RUN ${BUILD_ROOT}/buddy-mlir/llvm/build-cross-mlir-rv
RUN cmake -G Ninja ../../llvm/llvm \
     -DLLVM_ENABLE_PROJECTS="mlir" \
     -DLLVM_BUILD_EXAMPLES=OFF \
     -DCMAKE_CROSSCOMPILING=True \
     -DLLVM_TARGET_ARCH=RISCV64 \
     -DLLVM_TARGETS_TO_BUILD=RISCV \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DLLVM_NATIVE_ARCH=RISCV \
     -DLLVM_HOST_TRIPLE=riscv64-unknown-linux-gnu \
     -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
     -DCMAKE_C_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang \
     -DCMAKE_CXX_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang++ \
     -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
     -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
     -DMLIR_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-tblgen \
     -DLLVM_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/llvm-tblgen \
     -DMLIR_LINALG_ODS_YAML_GEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-linalg-ods-yaml-gen \
     -DMLIR_PDLL_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/mlir-pdll \
     -DLLVM_ENABLE_ZSTD=Off
RUN ninja
ENV BUILD_CROSS_MLIR_DIR=${BUILD_ROOT}/buddy-mlir/llvm/build-cross-mlir-rv


# 5. Build Cross-Compiled buddy-mlir
RUN mkdir ${BUILD_ROOT}/buddy-mlir/build-cross-rv
WORKDIR ${BUILD_ROOT}/buddy-mlir/build-cross-rv
RUN cmake -G Ninja .. \
     -DCMAKE_SYSTEM_NAME=Linux \
     -DMLIR_DIR=${BUILD_CROSS_MLIR_DIR}/lib/cmake/mlir \
     -DLLVM_DIR=${BUILD_CROSS_MLIR_DIR}/lib/cmake/llvm \
     -DCMAKE_CROSSCOMPILING=True \
     -DLLVM_TARGETS_TO_BUILD=RISCV \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DLLVM_NATIVE_ARCH=RISCV \
     -DLLVM_HOST_TRIPLE=riscv64-unknown-linux-gnu \
     -DCMAKE_C_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang \
     -DCMAKE_CXX_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang++ \
     -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
     -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
     -DLLVM_ENABLE_ZSTD=Off
RUN ninja StaticMLIRCRunnerUtils StaticMLIRRunnerUtils
