#include "perf_test.h"

extern void tanh(float* data, int channels, int size);

static void BM_Tanh(benchmark::State& state) {
    const int channels = state.range(0);
    const int size = state.range(1);

    std::vector<float> data = sequence_vector<float>(channels * size);

    for (auto _ : state) {
        tanh(data.data(), channels, size);

        benchmark::DoNotOptimize(data.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Tanh)
    ->Name("Tanh")
    ->ArgsProduct({

        {1024, 512, 512, 500, 256, 128},

        {128, 256, 512, 1024, 2048, 4096}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(500)
    ->Repetitions(3);
