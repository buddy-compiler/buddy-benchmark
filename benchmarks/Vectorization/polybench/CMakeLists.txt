add_executable(vectorization-polybench-benchmark Main.cpp)

target_link_libraries(vectorization-polybench-benchmark PRIVATE GoogleBenchmark)

separate_arguments(CLANG_FLAGS_LIST UNIX_COMMAND "${CMAKE_C_FLAGS}")

function(add_polybench_benchmark bench_name)
  set(mlir_file ${BUDDY_SOURCE_DIR}/benchmarks/Vectorization/polybench/${bench_name}.mlir)

  set(ll_file ${CMAKE_CURRENT_BINARY_DIR}/${bench_name}.ll)
  set(obj_file ${CMAKE_CURRENT_BINARY_DIR}/${bench_name}.o)

  add_custom_command(
    OUTPUT ${obj_file}
    COMMAND
      ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      ${mlir_file}
        -lower-affine
        -convert-scf-to-cf
        -expand-strided-metadata
        -convert-math-to-llvm
        -convert-arith-to-llvm
        -llvm-request-c-wrappers
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
      ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir -o ${ll_file}
    COMMAND
      ${LLVM_MLIR_BINARY_DIR}/clang -O0 ${CLANG_FLAGS_LIST} ${ll_file}
        -c -save-temps -o ${obj_file}
    DEPENDS ${mlir_file}
  )

  add_library(${bench_name} STATIC ${obj_file})
  set_target_properties(${bench_name} PROPERTIES LINKER_LANGUAGE CXX)

  target_sources(vectorization-polybench-benchmark PRIVATE ${bench_name}Benchmark.cpp)
  target_link_libraries(vectorization-polybench-benchmark PRIVATE ${bench_name})
endfunction()

add_polybench_benchmark(MLIRPolybench2mm)
add_polybench_benchmark(MLIRPolybench3mm)
add_polybench_benchmark(MLIRPolybenchAdi)
add_polybench_benchmark(MLIRPolybenchAtax)
add_polybench_benchmark(MLIRPolybenchBicg)
add_polybench_benchmark(MLIRPolybenchCholesky)
add_polybench_benchmark(MLIRPolybenchCorrelation)
add_polybench_benchmark(MLIRPolybenchCovariance)
add_polybench_benchmark(MLIRPolybenchDeriche)
add_polybench_benchmark(MLIRPolybenchDoitgen)
add_polybench_benchmark(MLIRPolybenchDurbin)
add_polybench_benchmark(MLIRPolybenchFdtd2D)
add_polybench_benchmark(MLIRPolybenchFloydWarshall)
add_polybench_benchmark(MLIRPolybenchGemm)
add_polybench_benchmark(MLIRPolybenchGemver)
add_polybench_benchmark(MLIRPolybenchGesummv)
add_polybench_benchmark(MLIRPolybenchGramschmidt)
add_polybench_benchmark(MLIRPolybenchHeat3D)
add_polybench_benchmark(MLIRPolybenchJacobi1D)
add_polybench_benchmark(MLIRPolybenchJacobi2D)
add_polybench_benchmark(MLIRPolybenchLu)
add_polybench_benchmark(MLIRPolybenchLudcmp)
add_polybench_benchmark(MLIRPolybenchMvt)
add_polybench_benchmark(MLIRPolybenchNussinov)
add_polybench_benchmark(MLIRPolybenchSeidel2D)
add_polybench_benchmark(MLIRPolybenchSymm)
add_polybench_benchmark(MLIRPolybenchSyr2k)
add_polybench_benchmark(MLIRPolybenchSyrk)
add_polybench_benchmark(MLIRPolybenchTrisolv)
add_polybench_benchmark(MLIRPolybenchTrmm)
