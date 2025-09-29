#include "perf_test.h"

extern void magnitude32f(const float* x, const float* y, float* mag, int len);

static void BM_Magnitude32f(benchmark::State& state) {
    const int n = state.range(0);

    auto x = sequence_vector<float>(n);
    auto y = sequence_vector<float>(n);
    std::vector<float> mag(n);

    for (auto _ : state) {
        magnitude32f(x.data(), y.data(), mag.data(), n);

        benchmark::DoNotOptimize(mag.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Magnitude32f)
    ->Name("Magnitude32f")
    ->RangeMultiplier(2)
    ->Range(65535, 1 << 20)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);
