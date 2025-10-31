#include "perf_test.h"

extern void SAXPY(const int n, const float* x, float* y, const float a);

static void BM_SAXPY(benchmark::State& state) {
  const int n = state.range(0);
  const float a = 3.14f;

  auto x = sequence_vector<float>(n);
  auto y = sequence_vector<float>(n);

  for (auto _ : state) {
    SAXPY(n, x.data(), y.data(), a);

    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_SAXPY)
    ->Name("SAXPY")
    ->RangeMultiplier(2)
    ->Range(65535, 1 << 20)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);