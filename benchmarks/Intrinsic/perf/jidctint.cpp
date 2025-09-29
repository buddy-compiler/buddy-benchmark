#include "perf_test.h"
#include <cstring>

extern void idct_islow(short *quantptr, short *coef_block, unsigned char **output_buf, unsigned int output_col, unsigned char* sample_range_limit);

typedef unsigned char JSAMPLE;
#define MAXJSAMPLE 255
#define CENTERJSAMPLE 128

static JSAMPLE * table;
static JSAMPLE * original_table_ptr;
static JSAMPLE * sample_range_limit;
static void setupTable() {
    int i;
    original_table_ptr = (JSAMPLE *)malloc((5 * (MAXJSAMPLE + 1) + CENTERJSAMPLE) * sizeof(JSAMPLE));
    table = original_table_ptr;
    table += (MAXJSAMPLE + 1);

    sample_range_limit = table;
    memset(table - (MAXJSAMPLE + 1), 0, (MAXJSAMPLE + 1) * sizeof(JSAMPLE));

    for (i = 0; i <= MAXJSAMPLE; i++)
    table[i] = (JSAMPLE)i;
    table += CENTERJSAMPLE;

    for (i = CENTERJSAMPLE; i < 2 * (MAXJSAMPLE + 1); i++)
    table[i] = MAXJSAMPLE;

    memset(table + (2 * (MAXJSAMPLE + 1)), 0,
        (2 * (MAXJSAMPLE + 1) - CENTERJSAMPLE) * sizeof(JSAMPLE));
    memcpy(table + (4 * (MAXJSAMPLE + 1) - CENTERJSAMPLE),
        sample_range_limit, CENTERJSAMPLE * sizeof(JSAMPLE));
}

static void BM_JpegIdctInteger(benchmark::State& state) {
    const int num_blocks = state.range(0);

    static bool table_initialized = false;
    if (!table_initialized) {
        setupTable();
        table_initialized = true;
    }

    std::vector<short> dct_blocks(num_blocks * 64);

    std::vector<short> quant_table(64);
    for (int i = 0; i < 64; i++) {
        quant_table[i] = 1 + (i % 8);
    }

    std::vector<std::vector<unsigned char>> output_rows(num_blocks);
    std::vector<std::vector<unsigned char*>> output_ptrs(num_blocks);

    for (int block = 0; block < num_blocks; block++) {
        output_rows[block].resize(8 * 8);
        output_ptrs[block].resize(8);
        for (int row = 0; row < 8; row++) {
            output_ptrs[block][row] = &output_rows[block][row * 8];
        }
    }

    for (int block = 0; block < num_blocks; block++) {
        for (int i = 0; i < 64; i++) {
            dct_blocks[block * 64 + i] = (short)((block * 8 + i * 5) % 1024 - 512);
        }
    }

    for (auto _ : state) {
        for (int block = 0; block < num_blocks; block++) {
            idct_islow(quant_table.data(), &dct_blocks[block * 64], output_ptrs[block].data(), 0, sample_range_limit);
        }
        benchmark::DoNotOptimize(output_rows[0].data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_blocks);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_blocks * 64);
}

BENCHMARK(BM_JpegIdctInteger)
    ->Name("JpegIdctInteger")
    ->RangeMultiplier(4)
    ->Range(64, 4096)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);
