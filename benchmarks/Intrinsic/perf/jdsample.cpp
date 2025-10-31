#include "perf_test.h"

extern void h2v1_upsample(int max_v_samp_factor, unsigned int output_width,
                          unsigned char **input_data, unsigned char ***output_data);

extern void h2v2_upsample(int max_v_samp_factor, unsigned int output_width,
                          unsigned char **input_data, unsigned char ***output_data);

extern void h2v1_fancy_upsample(int max_v_samp_factor, unsigned int downsampled_width,
                                unsigned char **input_data, unsigned char ***output_data);

extern void h2v2_fancy_upsample(int max_v_samp_factor, unsigned int downsampled_width,
                                unsigned char **input_data, unsigned char ***output_data);

static void BM_H2V1_Upsample(benchmark::State& state) {
    const int input_width = state.range(0) / 2;
    const int output_width = state.range(0);
    const int height = 2;

    std::vector<unsigned char*> input_data(height);
    for (int row = 0; row < height; row++) {
        input_data[row] = new unsigned char[input_width];
        for (int col = 0; col < input_width; col++) {
            input_data[row][col] = (unsigned char)(row * 64 + col * 16);
        }
    }

    std::vector<unsigned char*> output_data(height);
    for (int row = 0; row < height; row++) {
        output_data[row] = new unsigned char[output_width];
    }

    for (auto _ : state) {
        auto output_ptr = output_data.data();
        h2v1_upsample(height, output_width, input_data.data(), &output_ptr);
        benchmark::DoNotOptimize(output_data[0]);
    }

    for (int row = 0; row < height; row++) {
        delete[] input_data[row];
        delete[] output_data[row];
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * output_width * height);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * output_width * height);
}

static void BM_H2V2_Upsample(benchmark::State& state) {
    const int input_width = state.range(0) / 2;
    const int output_width = state.range(0);
    const int input_height = state.range(1) / 2;
    const int output_height = state.range(1);

    std::vector<unsigned char*> input_data(input_height);
    for (int row = 0; row < input_height; row++) {
        input_data[row] = new unsigned char[input_width];
        for (int col = 0; col < input_width; col++) {
            input_data[row][col] = (unsigned char)(row * 32 + col * 8);
        }
    }

    std::vector<unsigned char*> output_data(output_height);
    for (int row = 0; row < output_height; row++) {
        output_data[row] = new unsigned char[output_width];
    }

    for (auto _ : state) {
        auto output_ptr = output_data.data();
        h2v2_upsample(output_height, output_width, input_data.data(), &output_ptr);
        benchmark::DoNotOptimize(output_data[0]);
    }

    for (int row = 0; row < input_height; row++) {
        delete[] input_data[row];
    }
    for (int row = 0; row < output_height; row++) {
        delete[] output_data[row];
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * output_width * output_height);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * output_width * output_height);
}

static void BM_H2V1_Fancy_Upsample(benchmark::State& state) {
    const int downsampled_width = state.range(0) / 2;
    const int output_width = state.range(0);
    const int height = 2;

    std::vector<unsigned char*> input_data(height);
    for (int row = 0; row < height; row++) {
        input_data[row] = new unsigned char[downsampled_width];
        for (int col = 0; col < downsampled_width; col++) {
            input_data[row][col] = (unsigned char)(row * 64 + col * 16);
        }
    }

    std::vector<unsigned char*> output_data(height);
    for (int row = 0; row < height; row++) {
        output_data[row] = new unsigned char[output_width];
    }

    for (auto _ : state) {
        auto output_ptr = output_data.data();
        h2v1_fancy_upsample(height, downsampled_width, input_data.data(), &output_ptr);
        benchmark::DoNotOptimize(output_data[0]);
    }

    for (int row = 0; row < height; row++) {
        delete[] input_data[row];
        delete[] output_data[row];
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * output_width * height);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * output_width * height);
}

static void BM_H2V2_Fancy_Upsample(benchmark::State& state) {
    const int downsampled_width = state.range(0) / 2;
    const int output_width = state.range(0);
    const int input_height = state.range(1) / 2;
    const int output_height = state.range(1);

    const int context_rows = 2;
    std::vector<unsigned char*> input_data(input_height + context_rows);
    for (int row = 0; row < input_height + context_rows; row++) {
        input_data[row] = new unsigned char[downsampled_width];
        for (int col = 0; col < downsampled_width; col++) {
            input_data[row][col] = (unsigned char)(row * 32 + col * 8);
        }
    }

    std::vector<unsigned char*> output_data(output_height);
    for (int row = 0; row < output_height; row++) {
        output_data[row] = new unsigned char[output_width];
    }

    for (auto _ : state) {
        auto output_ptr = output_data.data();

        h2v2_fancy_upsample(output_height, downsampled_width,
                           input_data.data() + 1, &output_ptr);
        benchmark::DoNotOptimize(output_data[0]);
    }

    for (int row = 0; row < input_height + context_rows; row++) {
        delete[] input_data[row];
    }
    for (int row = 0; row < output_height; row++) {
        delete[] output_data[row];
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * output_width * output_height);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * output_width * output_height);
}

BENCHMARK(BM_H2V1_Upsample)
    ->Name("H2V1_Upsample")
    ->RangeMultiplier(2)
    ->Range(128, 2048)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_H2V2_Upsample)
    ->Name("H2V2_Upsample")
    ->ArgsProduct({
        {128, 256, 512, 1024, 2048},
        {128, 256, 512, 1024}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_H2V1_Fancy_Upsample)
    ->Name("H2V1_Fancy_Upsample")
    ->RangeMultiplier(2)
    ->Range(128, 2048)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_H2V2_Fancy_Upsample)
    ->Name("H2V2_Fancy_Upsample")
    ->ArgsProduct({
        {128, 256, 512, 1024, 2048},
        {128, 256, 512, 1024}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);
