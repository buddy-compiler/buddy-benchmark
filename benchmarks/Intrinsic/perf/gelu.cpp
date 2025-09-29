#include "perf_test.h"
#include <vector>
#include <cstdlib>

extern void gelu(float* data, int channels, int size, bool fast_gelu);

template <typename T>
std::vector<T> sequence_vector(int n)
{
    std::vector<T> v(n);
    for (int i = 0; i < n; i++)
    {
        v[i] = (T)(i % 10 - 5);
    }
    return v;
}

static void BM_GELU(benchmark::State& state) {
    const int channels = state.range(0);
    const int size = state.range(1);
    const bool fast = false;

    auto data = sequence_vector<float>(channels * size);

    for (auto _ : state) {
        gelu(data.data(), channels, size, fast);

        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
}

static void BM_FastGELU(benchmark::State& state) {
    const int channels = state.range(0);
    const int size = state.range(1);
    const bool fast = true;

    auto data = sequence_vector<float>(channels * size);

    for (auto _ : state) {
        gelu(data.data(), channels, size, fast);

        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_GELU)
    ->Name("GELU")
    ->Args({1, 1 << 17})
    ->Args({256, 512})
    ->Args({128, 2048})
    ->Args({64, 4096})
    ->Args({128, 8192})
    ->Args({256, 1024})
    ->Args({512, 1024})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(300)
    ->Repetitions(3);

BENCHMARK(BM_FastGELU)
    ->Name("FastGELU")
    ->Args({1, 1 << 17})
    ->Args({256, 512})
    ->Args({128, 2048})
    ->Args({64, 4096})
    ->Args({128, 8192})
    ->Args({256, 1024})
    ->Args({512, 1024})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(300)
    ->Repetitions(3);
