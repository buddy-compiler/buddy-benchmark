# add_subdirectory(Preprocesses)
add_subdirectory(Operations)

#-------------------------------------------------------------------------------
# Third-Party Sub Directory
#-------------------------------------------------------------------------------

add_subdirectory(${KFR_DIR} ./kfr)
include_directories(${KFR_DIR}/include)

#-------------------------------------------------------------------------------
# Optimization Configurations
#-------------------------------------------------------------------------------

if (${BUDDY_OPT_STRIP_MINING})
  set(SPLITING_SIZE ${BUDDY_OPT_STRIP_MINING} CACHE STRING "Spliting Size")
elseif(HAVE_AVX512)
  set(SPLITING_SIZE 256 CACHE STRING "Spliting Size")
elseif(HAVE_AVX2)
  set(SPLITING_SIZE 128 CACHE STRING "Spliting Size")
elseif(HAVE_SSE)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
elseif(HAVE_NEON)
  set(SPLITING_SIZE 64 CACHE STRING "Spliting Size")
else()  # Fallback value, avoid crashing
  set(SPLITING_SIZE 16 CACHE STRING "Spliting Size")
endif()

set(MLIR_LINALG_TILE 2 CACHE STRING "MLIR Linalg Tile Size")
set(LLVM_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin)

message(STATUS "Configuring Stride Size: ${SPLITING_SIZE}")
message(STATUS "Configuring MLIR Linalg Tile Size: ${MLIR_LINALG_TILE}")
