#include "perf_test.h"
#include <vector>
#include <cstdlib>
#include <cmath>

extern void selu(float* data, int channels, int size, float alpha = 1.673264f, float lambda = 1.050700f);

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

static void BM_SELU(benchmark::State& state) {
    const int channels = state.range(0);
    const int size = state.range(1);
    const float alpha = 1.673264f;
    const float lambda = 1.050700f;

    auto data = sequence_vector<float>(channels * size);

    for (auto _ : state) {
        selu(data.data(), channels, size, alpha, lambda);

        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_SELU)
    ->Name("SELU")

    ->Args({1, 1024})
    ->Args({1, 8192})

    ->Args({16, 1024})
    ->Args({32, 2048})
    ->Args({64, 4096})

    ->Args({64, 8192})
    ->Args({128, 4096})
    ->Args({128, 8192})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);
