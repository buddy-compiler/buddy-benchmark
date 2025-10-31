#include "perf_test.h"

extern void fdct_islow(short *data);

static void BM_JpegFdctInteger(benchmark::State& state) {
    const int num_blocks = state.range(0);

    std::vector<short> blocks(num_blocks * 64);

    for (int block = 0; block < num_blocks; block++) {
        for (int i = 0; i < 64; i++) {
            blocks[block * 64 + i] = (short)((block * 16 + i * 3) % 512 - 256);
        }
    }

    for (auto _ : state) {
        for (int block = 0; block < num_blocks; block++) {
            fdct_islow(&blocks[block * 64]);
        }
        benchmark::DoNotOptimize(blocks.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_blocks);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_blocks * 64 * sizeof(short));
}

BENCHMARK(BM_JpegFdctInteger)
    ->Name("JpegFdctInteger")
    ->RangeMultiplier(4)
    ->Range(64, 4096)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);
