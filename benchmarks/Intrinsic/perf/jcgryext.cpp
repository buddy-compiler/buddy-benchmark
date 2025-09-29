#include "perf_test.h"

extern void rgb_gray_convert(unsigned int image_width, unsigned char **input_buf,
                             unsigned char ***output_buf, unsigned int output_row, int num_rows);

static void BM_RgbGrayConvert(benchmark::State& state) {
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

    std::vector<unsigned char*> gray_plane(height);
    for (int row = 0; row < height; row++) {
        gray_plane[row] = new unsigned char[width];
    }

    unsigned char **gray_output[1] = {gray_plane.data()};

    for (auto _ : state) {
        rgb_gray_convert(width, rgb_data.data(), gray_output, 0, height);
        benchmark::DoNotOptimize(gray_plane[0]);
    }

    for (int row = 0; row < height; row++) {
        delete[] rgb_data[row];
        delete[] gray_plane[row];
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * width * height * 3);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * width * height);
}

BENCHMARK(BM_RgbGrayConvert)
    ->Name("RgbGrayConvert")
    ->ArgsProduct({
        {64, 128, 256, 512, 1024},
        {64, 128, 256, 512}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);
