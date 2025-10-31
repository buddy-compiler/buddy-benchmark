#include "perf_test.h"
#include <vector>

extern void swish(float* data, int channels, int size);

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

static void BM_Swish(benchmark::State& state) {
    const int channels = state.range(0);
    const int size = state.range(1);

    std::vector<float> data = sequence_vector<float>(channels * size);

    for (auto _ : state) {
        swish(data.data(), channels, size);

        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Swish)
    ->Name("Swish")
    ->ArgsProduct({

        {1024, 512, 256, 128, 64, 32},

        {128, 256, 512, 1024, 2048, 4096}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);
