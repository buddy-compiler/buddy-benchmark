#include "perf_test.h"

extern void rgb_ycc_convert(unsigned int image_width, unsigned char **input_buf,
                            unsigned char ***output_buf, unsigned int output_row, int num_rows);

static void BM_RgbYccConvert(benchmark::State& state) {
    const int width = state.range(0);
    const int height = state.range(1);

    std::vector<unsigned char*> rgb_data(height);
    for (int row = 0; row < height; row++) {
        rgb_data[row] = new unsigned char[width * 3];

        for (int col = 0; col < width; col++) {
            int base = col * 3;
            rgb_data[row][base + 0] = (unsigned char)(row * 64 + col * 8);
            rgb_data[row][base + 1] = (unsigned char)(row * 32 + col * 16);
            rgb_data[row][base + 2] = (unsigned char)(row * 16 + col * 32);
        }
    }

    std::vector<std::vector<unsigned char*>> ycc_planes(3);
    for (int plane = 0; plane < 3; plane++) {
        ycc_planes[plane].resize(height);
        for (int row = 0; row < height; row++) {
            ycc_planes[plane][row] = new unsigned char[width];
        }
    }

    unsigned char **ycc_output[3] = {
        ycc_planes[0].data(),
        ycc_planes[1].data(),
        ycc_planes[2].data()
    };

    for (auto _ : state) {
        rgb_ycc_convert(width, rgb_data.data(), ycc_output, 0, height);
        benchmark::DoNotOptimize(ycc_planes[0][0]);
        benchmark::DoNotOptimize(ycc_planes[1][0]);
        benchmark::DoNotOptimize(ycc_planes[2][0]);
    }

    for (int row = 0; row < height; row++) {
        delete[] rgb_data[row];
    }
    for (int plane = 0; plane < 3; plane++) {
        for (int row = 0; row < height; row++) {
            delete[] ycc_planes[plane][row];
        }
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * width * height * 3);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * width * height);
}

BENCHMARK(BM_RgbYccConvert)
    ->Name("RgbYccConvert")
    ->ArgsProduct({
        {64, 128, 256, 512, 1024},
        {64, 128, 256, 512}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);
