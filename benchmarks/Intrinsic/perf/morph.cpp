#include "perf_test.h"
#include <vector>
#include <cmath>

#define CV_HAL_MORPH_ERODE   0
#define CV_HAL_MORPH_DILATE  1
#define CV_8U                0

extern int morph(int op, int src_type, int dst_type,
                 uint8_t* src_data, size_t src_step,
                 uint8_t* dst_data, size_t dst_step,
                 int width, int height,
                 int roi_width, int roi_height, int roi_x, int roi_y,
                 int roi_width2, int roi_height2, int roi_x2, int roi_y2,
                 int kernel_type, uint8_t* kernel_data, size_t kernel_step,
                 int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                 int borderType, const double borderValue[4], int iterations, bool isSubmatrix);

static std::vector<uint8_t> generate_kernel_3x3(const std::string& type = "full") {
    if (type == "cross") {
        return {0, 1, 0,
                1, 1, 1,
                0, 1, 0};
    } else {

        return {1, 1, 1,
                1, 1, 1,
                1, 1, 1};
    }
}

template<typename T>
static std::vector<T> generate_test_image(int width, int height, const std::string& pattern = "gradient") {

    width = std::max(width, 5);
    height = std::max(height, 5);

    std::vector<T> img(width * height);

    if (pattern == "gradient") {

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img[y * width + x] = static_cast<T>((x + y) % 256);
            }
        }
    } else if (pattern == "checkerboard") {

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img[y * width + x] = ((x/8 + y/8) % 2) ? 255 : 0;
            }
        }
    } else if (pattern == "random") {

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img[y * width + x] = static_cast<T>((x * 73 + y * 137) % 256);
            }
        }
    } else {

        for (size_t i = 0; i < img.size(); i++) {
            img[i] = static_cast<T>(i % 256);
        }
    }

    return img;
}

static void BM_MorphErode3x3(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 5);
    height = std::max(height, 5);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "gradient");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("full");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    for (auto _ : state) {
        morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              width, height, 0, 0, width, height, 0, 0,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              0, borderValue, 1, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * actual_size);
    state.SetBytesProcessed(state.iterations() * actual_size * sizeof(uint8_t) * 2);
}

static void BM_MorphDilate3x3(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 5);
    height = std::max(height, 5);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "gradient");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("full");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    for (auto _ : state) {
        morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              width, height, 0, 0, width, height, 0, 0,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              0, borderValue, 1, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * actual_size);
    state.SetBytesProcessed(state.iterations() * actual_size * sizeof(uint8_t) * 2);
}

static void BM_MorphErodeCross3x3(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 5);
    height = std::max(height, 5);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "checkerboard");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("cross");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    for (auto _ : state) {
        morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              width, height, 0, 0, width, height, 0, 0,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              0, borderValue, 1, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * actual_size);
    state.SetBytesProcessed(state.iterations() * actual_size * sizeof(uint8_t) * 2);
}

static void BM_MorphDilateCross3x3(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 5);
    height = std::max(height, 5);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "checkerboard");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("cross");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    for (auto _ : state) {
        morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              width, height, 0, 0, width, height, 0, 0,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              0, borderValue, 1, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * actual_size);
    state.SetBytesProcessed(state.iterations() * actual_size * sizeof(uint8_t) * 2);
}

static void BM_MorphErodeReplicate3x3(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 5);
    height = std::max(height, 5);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "random");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("full");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    for (auto _ : state) {
        morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              width, height, 0, 0, width, height, 0, 0,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              4, borderValue, 1, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * actual_size);
    state.SetBytesProcessed(state.iterations() * actual_size * sizeof(uint8_t) * 2);
}

static void BM_MorphDilateReplicate3x3(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 5);
    height = std::max(height, 5);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "random");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("full");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    for (auto _ : state) {
        morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              width, height, 0, 0, width, height, 0, 0,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              4, borderValue, 1, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * actual_size);
    state.SetBytesProcessed(state.iterations() * actual_size * sizeof(uint8_t) * 2);
}

static void BM_MorphErodeROI(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 10);
    height = std::max(height, 10);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "gradient");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("full");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    int roi_width = width - 4;
    int roi_height = height - 4;
    int roi_x = 2;
    int roi_y = 2;

    for (auto _ : state) {
        morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              roi_width, roi_height, roi_x, roi_y, roi_width, roi_height, roi_x, roi_y,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              0, borderValue, 1, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * roi_width * roi_height);
    state.SetBytesProcessed(state.iterations() * roi_width * roi_height * sizeof(uint8_t) * 2);
}

static void BM_MorphErodeMultipleIterations(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 5);
    height = std::max(height, 5);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "gradient");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("full");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    for (auto _ : state) {
        morph(CV_HAL_MORPH_ERODE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              width, height, 0, 0, width, height, 0, 0,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              0, borderValue, 3, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * actual_size * 3);
    state.SetBytesProcessed(state.iterations() * actual_size * sizeof(uint8_t) * 2);
}

static void BM_MorphDilateMultipleIterations(benchmark::State& state) {
    const int size = state.range(0);
    int width = static_cast<int>(std::sqrt(size));
    int height = size / width;

    width = std::max(width, 5);
    height = std::max(height, 5);
    const int actual_size = width * height;

    auto src = generate_test_image<uint8_t>(width, height, "gradient");
    std::vector<uint8_t> dst(actual_size);
    auto kernel = generate_kernel_3x3("full");
    double borderValue[4] = {0.0, 0.0, 0.0, 0.0};

    for (auto _ : state) {
        morph(CV_HAL_MORPH_DILATE, CV_8U, CV_8U,
              src.data(), width, dst.data(), width, width, height,
              width, height, 0, 0, width, height, 0, 0,
              CV_8U, kernel.data(), 3, 3, 3, 1, 1,
              0, borderValue, 2, false);

        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * actual_size * 2);
    state.SetBytesProcessed(state.iterations() * actual_size * sizeof(uint8_t) * 2);
}

BENCHMARK(BM_MorphErode3x3)
    ->Name("MorphErode3x3_Full")
    ->RangeMultiplier(4)
    ->Range(25, 1 << 18)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_MorphDilate3x3)
    ->Name("MorphDilate3x3_Full")
    ->RangeMultiplier(4)
    ->Range(25, 1 << 18)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_MorphErodeCross3x3)
    ->Name("MorphErode3x3_Cross")
    ->RangeMultiplier(4)
    ->Range(25, 1 << 18)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_MorphDilateCross3x3)
    ->Name("MorphDilate3x3_Cross")
    ->RangeMultiplier(4)
    ->Range(25, 1 << 18)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_MorphErodeReplicate3x3)
    ->Name("MorphErode3x3_Replicate")
    ->RangeMultiplier(4)
    ->Range(25, 1 << 18)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_MorphDilateReplicate3x3)
    ->Name("MorphDilate3x3_Replicate")
    ->RangeMultiplier(4)
    ->Range(25, 1 << 18)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(100)
    ->Repetitions(3);

BENCHMARK(BM_MorphErodeROI)
    ->Name("MorphErode3x3_ROI")
    ->RangeMultiplier(4)
    ->Range(100, 1 << 18)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_MorphErodeMultipleIterations)
    ->Name("MorphErode3x3_3Iterations")
    ->RangeMultiplier(4)
    ->Range(25, 1 << 16)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(50)
    ->Repetitions(3);

BENCHMARK(BM_MorphDilateMultipleIterations)
    ->Name("MorphDilate3x3_2Iterations")
    ->RangeMultiplier(4)
    ->Range(25, 1 << 16)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(50)
    ->Repetitions(3);
