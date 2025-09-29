set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED TOOLCHAIN_COMPILER_LOCATION_HINT)
  set(TOOLCHAIN_COMPILER_LOCATION_HINT /opt/riscv/bin)
endif()

find_program(CMAKE_C_COMPILER NAMES riscv64-unknown-linux-gnu-gcc PATHS ${TOOLCHAIN_COMPILER_LOCATION_HINT})
find_program(CMAKE_CXX_COMPILER NAMES riscv64-unknown-linux-gnu-g++ PATHS ${TOOLCHAIN_COMPILER_LOCATION_HINT})
find_program(CMAKE_LINKER NAMES riscv64-unknown-linux-gnu-ld PATHS ${TOOLCHAIN_COMPILER_LOCATION_HINT})

if(NOT DEFINED RISCV_SYSROOT)
  get_filename_component(_base_dir ${CMAKE_C_COMPILER} DIRECTORY)
  get_filename_component(_base_dir ${_base_dir} DIRECTORY)
  set(RISCV_SYSROOT ${_base_dir}/sysroot CACHE PATH "RISC-V sysroot")
endif()

set(CMAKE_SYSROOT "${RISCV_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${RISCV_SYSROOT})