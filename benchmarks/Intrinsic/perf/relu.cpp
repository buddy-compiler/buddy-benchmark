#include "perf_test.h"
#include <vector>
#include <cstdlib>

extern void relu(float* data, int channels, int size, float slope = 0.f);

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

static void BM_ReLU(benchmark::State& state) {
    const int channels = state.range(0);
    const int size = state.range(1);
    const float slope = 0.1f;

    auto data = sequence_vector<float>(channels * size);

    for (auto _ : state) {
        relu(data.data(), channels, size, slope);

        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_ReLU)
    ->Name("ReLU")
    ->Args({1, 1 << 17})
    ->Args({256, 512})
    ->Args({128, 2048})
    ->Args({64, 4096})
    ->Args({128, 8192})
    ->Args({256, 1024})
    ->Args({512, 1024})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);