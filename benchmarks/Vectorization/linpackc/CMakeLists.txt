#-------------------------------------------------------------------------------
# MLIR SCF Dialect linpackc Operation + Upstream Lowering Passes
#-------------------------------------------------------------------------------
include_directories(${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/linpackc)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
link_directories(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
add_subdirectory(Daxpy)
add_subdirectory(Ddot)
add_subdirectory(Dmxpy)
add_subdirectory(Dscal)
add_subdirectory(Epslon)
add_subdirectory(Idamax)
add_subdirectory(Matgen)
add_subdirectory(Dgefa)
add_subdirectory(Dgesl)

#-------------------------------------------------------------------------------
# Vectorization Benchmark Targets
#-------------------------------------------------------------------------------

add_executable(vectorization-linpackc-benchmark
  Main.cpp
  ./Daxpy/MLIRLinpackCDaxpyBenchmark.cpp
  ./Matgen/MLIRLinpackCMatgenBenchmark.cpp
  ./Ddot/MLIRLinpackCDdotBenchmark.cpp
  ./Dscal/MLIRLinpackCDscalBenchmark.cpp
  ./Idamax/MLIRLinpackCIdamaxBenchmark.cpp
  ./Dmxpy/MLIRLinpackCDmxpyBenchmark.cpp
  ./Epslon/MLIRLinpackCEpslonBenchmark.cpp
  ./Dgefa/MLIRLinpackCDgefaBenchmark.cpp
  ./Dgesl/MLIRLinpackCDgeslBenchmark.cpp
  )

target_link_libraries(vectorization-linpackc-benchmark
  GoogleBenchmark

  MLIRLinpackDaxpyf32
  MLIRLinpackDaxpyf64
  Daxpyfloat_gcc
  Daxpydouble_gcc
  Daxpyfloat_clang
  Daxpydouble_clang

  MLIRLinpackMatgenf32
  MLIRLinpackMatgenf64
  Matgenfloat_gcc
  Matgendouble_gcc
  Matgenfloat_clang
  Matgendouble_clang

  MLIRLinpackDdotf32
  MLIRLinpackDdotf64
  Ddotfloat_gcc
  Ddotdouble_gcc
  Ddotfloat_clang
  Ddotdouble_clang

  MLIRLinpackDscalf32
  MLIRLinpackDscalf64
  Dscalfloat_gcc
  Dscaldouble_gcc
  Dscalfloat_clang
  Dscaldouble_clang

  MLIRLinpackIdamaxf32
  MLIRLinpackIdamaxf64
  Idamaxfloat_gcc
  Idamaxdouble_gcc
  Idamaxfloat_clang
  Idamaxdouble_clang
  
  MLIRLinpackDmxpyf32
  MLIRLinpackDmxpyf64
  Dmxpyfloat_gcc
  Dmxpydouble_gcc
  Dmxpyfloat_clang
  Dmxpydouble_clang
  
  MLIRLinpackEpslonf32
  MLIRLinpackEpslonf64
  Epslonfloat_gcc
  Epslondouble_gcc
  Epslonfloat_clang
  Epslondouble_clang
  
  MLIRLinpackDgefaf32
  MLIRLinpackDgefaf64
  Dgefafloat_gcc
  Dgefadouble_gcc
  Dgefafloat_clang
  Dgefadouble_clang

  MLIRLinpackDgeslf32
  MLIRLinpackDgeslf64
  Dgeslfloat_gcc
  Dgesldouble_gcc
  Dgeslfloat_clang
  Dgesldouble_clang
  )
