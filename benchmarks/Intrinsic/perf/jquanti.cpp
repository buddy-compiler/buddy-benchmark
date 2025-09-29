#include "perf_test.h"

extern void quantize(short *coef_block, short *divisors, short *workspace);

static void BM_JpegQuantize(benchmark::State& state) {
    const int num_blocks = state.range(0);

    short quant_table[64] = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };

    std::vector<short> dct_blocks(num_blocks * 64);
    std::vector<short> workspace(num_blocks * 64);

    for (int block = 0; block < num_blocks; block++) {
        for (int i = 0; i < 64; i++) {
            dct_blocks[block * 64 + i] = (short)((block * 16 + i * 8) % 1024 - 512);
        }
    }

    for (auto _ : state) {
        for (int block = 0; block < num_blocks; block++) {
            quantize(&dct_blocks[block * 64], quant_table, &workspace[block * 64]);
        }
        benchmark::DoNotOptimize(workspace.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_blocks);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_blocks * 64 * sizeof(short));
}

BENCHMARK(BM_JpegQuantize)
    ->Name("JpegQuantize")
    ->RangeMultiplier(4)
    ->Range(64, 4096)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

extern void convsamp(unsigned char **sample_data, unsigned int start_col, short *workspace);

static void BM_JpegConvSamp(benchmark::State& state) {
    const int num_blocks = state.range(0);

    const int block_width = 8 + 4;
    std::vector<unsigned char> sample_buffer(num_blocks * 8 * block_width);
    std::vector<unsigned char*> sample_rows(num_blocks * 8);

    for (int block = 0; block < num_blocks; block++) {
        for (int row = 0; row < 8; row++) {
            sample_rows[block * 8 + row] = &sample_buffer[block * 8 * block_width + row * block_width];

            for (int col = 0; col < block_width; col++) {
                sample_rows[block * 8 + row][col] =
                    (unsigned char)((block * 16 + row * 8 + col * 4) % 256);
            }
        }
    }

    std::vector<short> workspace(num_blocks * 64);

    for (auto _ : state) {
        for (int block = 0; block < num_blocks; block++) {
            convsamp(&sample_rows[block * 8], 0, &workspace[block * 64]);
        }
        benchmark::DoNotOptimize(workspace.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_blocks);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_blocks * 64 * sizeof(unsigned char));
}

static void BM_JpegConvSampWithOffset(benchmark::State& state) {
    const int num_blocks = state.range(0);
    const unsigned int start_col = 2;

    const int block_width = 8 + 4;
    std::vector<unsigned char> sample_buffer(num_blocks * 8 * block_width);
    std::vector<unsigned char*> sample_rows(num_blocks * 8);

    for (int block = 0; block < num_blocks; block++) {
        for (int row = 0; row < 8; row++) {
            sample_rows[block * 8 + row] = &sample_buffer[block * 8 * block_width + row * block_width];

            for (int col = 0; col < block_width; col++) {
                sample_rows[block * 8 + row][col] =
                    ((row + col) % 2 == 0) ? 0 : 255;
            }
        }
    }

    std::vector<short> workspace(num_blocks * 64);

    for (auto _ : state) {
        for (int block = 0; block < num_blocks; block++) {
            convsamp(&sample_rows[block * 8], start_col, &workspace[block * 64]);
        }
        benchmark::DoNotOptimize(workspace.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_blocks);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_blocks * 64 * sizeof(unsigned char));
}

static void BM_JpegConvSampPatterns(benchmark::State& state) {
    const int num_blocks = state.range(0);
    const int pattern_type = state.range(1);

    const int block_width = 8 + 4;
    std::vector<unsigned char> sample_buffer(num_blocks * 8 * block_width);
    std::vector<unsigned char*> sample_rows(num_blocks * 8);

    for (int block = 0; block < num_blocks; block++) {
        for (int row = 0; row < 8; row++) {
            sample_rows[block * 8 + row] = &sample_buffer[block * 8 * block_width + row * block_width];

            for (int col = 0; col < block_width; col++) {
                unsigned char value;
                switch (pattern_type) {
                case 0:
                    value = 0;
                    break;
                case 1:
                    value = 255;
                    break;
                case 2:
                    value = (unsigned char)((row * 73 + col * 97 + block * 23) % 256);
                    break;
                case 3:
                default:
                    value = (unsigned char)((row * 32 + col * 16) % 256);
                    break;
                }
                sample_rows[block * 8 + row][col] = value;
            }
        }
    }

    std::vector<short> workspace(num_blocks * 64);

    for (auto _ : state) {
        for (int block = 0; block < num_blocks; block++) {
            convsamp(&sample_rows[block * 8], 0, &workspace[block * 64]);
        }
        benchmark::DoNotOptimize(workspace.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_blocks);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num_blocks * 64 * sizeof(unsigned char));
}

BENCHMARK(BM_JpegConvSamp)
    ->Name("JpegConvSamp")
    ->RangeMultiplier(4)
    ->Range(64, 4096)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_JpegConvSampWithOffset)
    ->Name("JpegConvSampWithOffset")
    ->RangeMultiplier(4)
    ->Range(64, 4096)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_JpegConvSampPatterns)
    ->Name("JpegConvSampPatterns")
    ->ArgsProduct({
        benchmark::CreateRange(64, 1024, 4),
        {0, 1, 2, 3}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(50)
    ->Repetitions(3);
