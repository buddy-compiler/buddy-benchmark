#include "perf_test.h"

extern void h2v1_merged_upsample(unsigned int output_width, unsigned char ***input_buf,
                                 unsigned int input_row, unsigned char **output_buf);

extern void h2v2_merged_upsample(unsigned int output_width, unsigned char ***input_buf,
                                 unsigned int input_row, unsigned char **output_buf);

static void BM_H2V1_MergedUpsample(benchmark::State& state) {
    const int input_width = state.range(0) / 2;
    const int output_width = state.range(0);
    const int height = 4;

    std::vector<std::vector<unsigned char*>> ycc_planes(3);
    for (int plane = 0; plane < 3; plane++) {
        ycc_planes[plane].resize(height);
        for (int row = 0; row < height; row++) {
            ycc_planes[plane][row] = new unsigned char[input_width];
            for (int col = 0; col < input_width; col++) {
                if (plane == 0) {
                    ycc_planes[plane][row][col] = (unsigned char)(row * 32 + col * 16);
                } else {
                    ycc_planes[plane][row][col] = 128;
                }
            }
        }
    }

    std::vector<unsigned char*> rgb_data(height);
    for (int row = 0; row < height; row++) {
        rgb_data[row] = new unsigned char[output_width * 3];
    }

    unsigned char **ycc_input[3] = {
        ycc_planes[0].data(),
        ycc_planes[1].data(),
        ycc_planes[2].data()
    };

    for (auto _ : state) {
        h2v1_merged_upsample(output_width, ycc_input, 0, rgb_data.data());
        benchmark::DoNotOptimize(rgb_data[0]);
    }

    for (int plane = 0; plane < 3; plane++) {
        for (int row = 0; row < height; row++) {
            delete[] ycc_planes[plane][row];
        }
    }
    for (int row = 0; row < height; row++) {
        delete[] rgb_data[row];
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * output_width * height * 3);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * output_width * height);
}

static void BM_H2V2_MergedUpsample(benchmark::State& state) {
    const int input_width = state.range(0) / 2;
    const int output_width = state.range(0);
    const int input_height = state.range(1) / 2;
    const int output_height = state.range(1);

    std::vector<std::vector<unsigned char*>> ycc_planes(3);
    for (int plane = 0; plane < 3; plane++) {
        ycc_planes[plane].resize(input_height);
        for (int row = 0; row < input_height; row++) {
            ycc_planes[plane][row] = new unsigned char[input_width];
            for (int col = 0; col < input_width; col++) {
                if (plane == 0) {
                    ycc_planes[plane][row][col] = (unsigned char)(row * 32 + col * 8);
                } else {
                    ycc_planes[plane][row][col] = 128;
                }
            }
        }
    }

    std::vector<unsigned char*> rgb_data(output_height);
    for (int row = 0; row < output_height; row++) {
        rgb_data[row] = new unsigned char[output_width * 3];
    }

    unsigned char **ycc_input[3] = {
        ycc_planes[0].data(),
        ycc_planes[1].data(),
        ycc_planes[2].data()
    };

    for (auto _ : state) {
        h2v2_merged_upsample(output_width, ycc_input, 0, rgb_data.data());
        benchmark::DoNotOptimize(rgb_data[0]);
    }

    for (int plane = 0; plane < 3; plane++) {
        for (int row = 0; row < input_height; row++) {
            delete[] ycc_planes[plane][row];
        }
    }
    for (int row = 0; row < output_height; row++) {
        delete[] rgb_data[row];
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * output_width * output_height * 3);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * output_width * output_height);
}

BENCHMARK(BM_H2V1_MergedUpsample)
    ->Name("H2V1_MergedUpsample")
    ->RangeMultiplier(2)
    ->Range(128, 2048)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_H2V2_MergedUpsample)
    ->Name("H2V2_MergedUpsample")
    ->ArgsProduct({
        {128, 256, 512, 1024, 2048},
        {128, 256, 512, 1024}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);
