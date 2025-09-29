#include "perf_test.h"

extern void fastAtan32f(const float* y, const float* x, float* dst, size_t n, bool angle_in_deg);

static void BM_FastAtan32f(benchmark::State& state) {
    const int n = state.range(0);
    const bool deg = (state.range(1) == 0);
    std::vector<float> Y = sequence_vector<float>(n);
    std::vector<float> X = sequence_vector<float>(n);
    std::vector<float> angles(n);

    for (auto _ : state) {
        fastAtan32f(Y.data(), X.data(), angles.data(), n, deg);
        benchmark::DoNotOptimize(angles.data());
    }
}

BENCHMARK(BM_FastAtan32f)
    ->Name("FastAtan32f")
    ->ArgsProduct({
        benchmark::CreateRange(65535, 1 << 20, 2),
        {0, 1}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);