#include "perf_test.h"

extern void h2v1_downsample(unsigned int image_width, int max_v_samp_factor,
                            unsigned int v_samp_factor, unsigned int width_in_blocks,
                            unsigned char **input_data, unsigned char **output_data);

extern void h2v2_downsample(unsigned int image_width, int max_v_samp_factor,
                            unsigned int v_samp_factor, unsigned int width_in_blocks,
                            unsigned char **input_data, unsigned char **output_data);

static void BM_H2V1_Downsample(benchmark::State& state) {
    const int width = state.range(0);
    const int height = 2;
    const int blocks_width = width / 16;

    std::vector<unsigned char*> input_data(height);
    for (int row = 0; row < height; row++) {
        input_data[row] = new unsigned char[width + 16];
        for (int col = 0; col < width; col++) {
            input_data[row][col] = (unsigned char)(row * 64 + col * 8);
        }
    }

    int out_width = width / 2;
    std::vector<unsigned char*> output_data(1);
    output_data[0] = new unsigned char[out_width];

    for (auto _ : state) {
        h2v1_downsample(width, height, 1, blocks_width, input_data.data(), output_data.data());
        benchmark::DoNotOptimize(output_data[0]);
    }

    for (int row = 0; row < height; row++) {
        delete[] input_data[row];
    }
    delete[] output_data[0];

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * width * height);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * out_width);
}

static void BM_H2V2_Downsample(benchmark::State& state) {
    const int width = state.range(0);
    const int height = state.range(1) * 2;
    const int blocks_width = width / 16;

    std::vector<unsigned char*> input_data(height);
    for (int row = 0; row < height; row++) {
        input_data[row] = new unsigned char[width + 16];
        for (int col = 0; col < width; col++) {
            input_data[row][col] = (unsigned char)(row * 32 + col * 4);
        }
    }

    int out_height = height / 2;
    int out_width = width / 2;
    std::vector<unsigned char*> output_data(out_height);
    for (int row = 0; row < out_height; row++) {
        output_data[row] = new unsigned char[out_width];
    }

    for (auto _ : state) {
        h2v2_downsample(width, height, out_height, blocks_width, input_data.data(), output_data.data());
        benchmark::DoNotOptimize(output_data[0]);
    }

    for (int row = 0; row < height; row++) {
        delete[] input_data[row];
    }
    for (int row = 0; row < out_height; row++) {
        delete[] output_data[row];
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * width * height);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * out_width * out_height);
}

BENCHMARK(BM_H2V1_Downsample)
    ->Name("H2V1_Downsample")
    ->RangeMultiplier(2)
    ->Range(64, 1024)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_H2V2_Downsample)
    ->Name("H2V2_Downsample")
    ->ArgsProduct({
        {64, 128, 256, 512, 1024},
        {32, 64, 128, 256}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);
