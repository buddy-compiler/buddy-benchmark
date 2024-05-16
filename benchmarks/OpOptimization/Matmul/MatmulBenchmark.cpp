//===- MatmulOpsBenchmark.cpp
//------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

namespace {
extern "C" {

void _mlir_ciface_matmul(MemRef<float, 2> *A, MemRef<float, 2> *B,
                         MemRef<float, 2> *C);
void _mlir_ciface_matmul_vector_8(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                  MemRef<float, 2> *C);
void _mlir_ciface_matmul_vector_16(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
void _mlir_ciface_matmul_vector_32(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
void _mlir_ciface_matmul_vector_64(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
void _mlir_ciface_matmul_vector_128(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                    MemRef<float, 2> *C);
}

#define DefineMatmulBenchmark(name, func)                                      \
  void name(benchmark::State &state) {                                         \
    int M = 256, N = 256, K = 256;                                             \
    intptr_t sizesA[2] = {M, K};                                               \
    intptr_t sizesB[2] = {K, N};                                               \
    intptr_t sizesC[2] = {M, N};                                               \
    MemRef<float, 2> A(sizesA, 1.0);                                           \
    MemRef<float, 2> B(sizesB, 1.0);                                           \
    MemRef<float, 2> C(sizesC, 0.0);                                           \
    for (auto _ : state) {                                                     \
      func(&A, &B, &C);                                                        \
    }                                                                          \
  }

#define RUN_BENCHMARK(name, func)                                              \
  Define##name##Benchmark(name, func) BENCHMARK(name)->Unit(                   \
      benchmark::kMillisecond);                                                \
  Define##name##Benchmark(name##Vector8, func##_vector_8)                      \
      BENCHMARK(name##Vector8)                                                 \
          ->Unit(benchmark::kMillisecond);                                     \
  Define##name##Benchmark(name##Vector16, func##_vector_16)                    \
      BENCHMARK(name##Vector16)                                                \
          ->Unit(benchmark::kMillisecond);                                     \
  Define##name##Benchmark(name##Vector32, func##_vector_32)                    \
      BENCHMARK(name##Vector32)                                                \
          ->Unit(benchmark::kMillisecond);                                     \
  Define##name##Benchmark(name##Vector64, func##_vector_64)                    \
      BENCHMARK(name##Vector64)                                                \
          ->Unit(benchmark::kMillisecond);                                     \
  Define##name##Benchmark(name##Vector128, func##_vector_128)                  \
      BENCHMARK(name##Vector128)                                               \
          ->Unit(benchmark::kMillisecond);

RUN_BENCHMARK(Matmul, _mlir_ciface_matmul)

} // namespace
