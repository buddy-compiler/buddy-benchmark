#include "perf_test.h"
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>

extern void hardswish(float* data, int channels, int size, float alpha = 0.2f, float beta = 0.5f);

template <typename T>
std::vector<T> sequence_vector(int n)
{
    std::vector<T> v(n);
    for (int i = 0; i < n; i++)
    {
        v[i] = (T)((i % 20) - 10);
    }
    return v;
}

static void BM_HardSwish(benchmark::State& state) {
    const int channels = state.range(0);
    const int size = state.range(1);
    const float alpha = 0.2f;
    const float beta = 0.5f;

    auto data = sequence_vector<float>(channels * size);

    for (auto _ : state) {
        hardswish(data.data(), channels, size, alpha, beta);

        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_HardSwish)
    ->Name("HardSwish")

    ->Args({1, 1024})
    ->Args({1, 8192})

    ->Args({16, 1024})
    ->Args({32, 2048})
    ->Args({64, 4096})

    ->Args({32, 8192})
    ->Args({64, 4096})
    ->Args({128, 4096})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);
