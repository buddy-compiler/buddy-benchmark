# Prepare RISC-V OpenMP ToolChain

In "Cross Compile to Target Platform" in [Deep Learning Benchmark](../benchmarks/DeepLearning/README.md), you can use either method as follows to enable the openmp feature on RISC-V.

## Directly Download

You can download "build-omp-shared-rv" from [this link](https://drive.google.com/file/d/1XEsAhOcMioN9gdufuyO9OrHIdR0UtHh2/view) and place it under `${LLVM_MLIR_BUILD_DIR}/../` to get environment variable `RISCV_OMP_SHARED` ready. See the following steps:
```
$ cd ${LLVM_MLIR_BUILD_DIR}/../
$ wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XEsAhOcMioN9gdufuyO9OrHIdR0UtHh2' -O build-omp-shared-rv.tar.gz`
$ tar -xzf build-omp-shared-rv.tar.gz
$ export RISCV_OMP_SHARED=${LLVM_MLIR_BUILD_DIR}/../build-omp-shared-rv/libomp.so
```

## Build From Source Code
In the step "Build Local LLVM/MLIR" in [Environment Setup Guide for MLIR and RVV Testing and Experiments](https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/RVVEnvironment.md), the configuration `-DLLVM_ENABLE_PROJECTS="mlir;clang;openmp"` only help build OpenMP on X86. If you want to build OpenMP on RV from source code, you need to make the following changes:

**STEP ONE:** Modify the ***corresponding part*** in `llvm/openmp/runtime/src/CMakeLists.txt` like this:
```
libomp_get_libflags(LIBOMP_CONFIGURED_LIBFLAGS)
# Build libomp library. Add LLVMSupport dependency if building in-tree with libomptarget profiling enabled.
if(OPENMP_STANDALONE_BUILD OR (NOT OPENMP_ENABLE_LIBOMP_PROFILING))
add_library(omp ${LIBOMP_LIBRARY_KIND} ${LIBOMP_SOURCE_FILES})
target_compile_options(omp PRIVATE ${CMAKE_C_FLAGS}) # new added
# Linking command will include libraries in LIBOMP_CONFIGURED_LIBFLAGS
target_link_libraries(omp ${LIBOMP_CONFIGURED_LIBFLAGS} ${LIBOMP_DL_LIBS})
```


**STEP TWO:** Use the following configuration to cross compile openmp with clang.
```
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="RISCV" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_C_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang \
    -DCMAKE_CXX_COMPILER=${BUILD_LOCAL_LLVM_DIR}/bin/clang++ \
    -DCMAKE_C_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DCMAKE_CXX_FLAGS="--target=riscv64-unknown-linux-gnu --sysroot=${RISCV_GNU_TOOLCHAIN_SYSROOT_DIR} --gcc-toolchain=${BUILD_RISCV_GNU_TOOLCHAIN_DIR}" \
    -DLLVM_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/llvm-tblgen \
    -DCLANG_TABLEGEN=${BUILD_LOCAL_LLVM_DIR}/bin/clang-tblgen \
    -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-linux-gnu \
    -DLLVM_TARGET_ARCH=RISCV64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENMP_ENABLE_LIBOMPTARGET=Off \
	-DLIBOMP_ARCH=riscv64 \
	-DLIBOMP_ENABLE_SHARED=On

ninja  clang lli omp
```
