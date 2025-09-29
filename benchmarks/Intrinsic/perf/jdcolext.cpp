#include "perf_test.h"

extern void ycc_rgb_convert(unsigned int output_width, unsigned char ***input_buf,
                            unsigned int input_row, unsigned char **output_buf, int num_rows);

static void BM_YccRgbConvert(benchmark::State& state) {
    const int width = state.range(0);
    const int height = state.range(1);

    std::vector<std::vector<unsigned char*>> ycc_planes(3);
    for (int plane = 0; plane < 3; plane++) {
        ycc_planes[plane].resize(height);
        for (int row = 0; row < height; row++) {
            ycc_planes[plane][row] = new unsigned char[width];

            for (int col = 0; col < width; col++) {
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
        rgb_data[row] = new unsigned char[width * 3];
    }

    unsigned char **ycc_input[3] = {
        ycc_planes[0].data(),
        ycc_planes[1].data(),
        ycc_planes[2].data()
    };

    for (auto _ : state) {
        ycc_rgb_convert(width, ycc_input, 0, rgb_data.data(), height);
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

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * width * height * 3);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * width * height);
}

BENCHMARK(BM_YccRgbConvert)
    ->Name("YccRgbConvert")
    ->ArgsProduct({
        {64, 128, 256, 512, 1024},
        {64, 128, 256, 512}
    })
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);
