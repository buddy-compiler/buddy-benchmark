##===- riscv-toolchain.cmake ----------------------------------------------===//
##
## Cross-compile benchmarks for RISC-V.
##
##===----------------------------------------------------------------------===//

if(RISCV_TOOLCHAIN_INCLUDED)
  return()
endif(RISCV_TOOLCHAIN_INCLUDED)
set(RISCV_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Set RISC-V toolchain path.
set(RISCV_TOOLCHAIN_ROOT "/root/riscv-gnu-toolchain/install")
set(CMAKE_FIND_ROOT_PATH ${RISCV_TOOLCHAIN_ROOT})
list(APPEND CMAKE_PREFIX_PATH ${RISCV_TOOLCHAIN_ROOT})
set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++")

# Comfigure RISC-V compiler flags.
set(RISCV_COMPILER_FLAGS "-march=rv64i2p0ma2p0f2p0d2p0c2p0 -mabi=lp64d")
set(CMAKE_C_FLAGS "${RISCV_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${RISCV_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
