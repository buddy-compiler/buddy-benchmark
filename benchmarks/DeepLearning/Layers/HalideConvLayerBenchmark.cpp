#include <chrono>
#include <cstdio>

#include "conv_layer_nonschedule.h"
#include "conv_layer_manuallyschedule.h"
#include "conv_layer_autoschedule.h"
#include <benchmark/benchmark.h>
#include "HalideBuffer.h"

using namespace Halide::Runtime;

const int N = 5, CI = 128, CO = 128, W = 100, H = 80;

Buffer<float, 4> input(CI, W + 2, H + 2, N), input1(CI, W + 2, H + 2, N), input2(CI, W + 2, H + 2, N);
Buffer<float, 4> filter(CO, 3, 3, CI), filter1(CO, 3, 3, CI), filter2(CO, 3, 3, CI);
Buffer<float, 1> bias(CO), bias1(CO), bias2(CO);
Buffer<float, 4> output(CO, W, H, N), output1(CO, W, H, N), output2(CO, W, H, N);

void initializeHalideConvLayerBenchmark(char **argv) {
    for (int c = 0; c < input.dim(3).extent(); c++) {
        for (int z = 0; z < input.channels(); z++) {
            for (int y = 0; y < input.height(); y++) {
                for (int x = 0; x < input.width(); x++) {
                    input(x, y, z, c) = rand();
                    input1(x, y, z, c) = input(x, y, z, c);
                    input2(x, y, z, c) = input(x, y, z, c);
                }
            }
        }
    }

    for (int c = 0; c < filter.dim(3).extent(); c++) {
        for (int z = 0; z < filter.channels(); z++) {
            for (int y = 0; y < filter.height(); y++) {
                for (int x = 0; x < filter.width(); x++) {
                    filter(x, y, z, c) = rand();
                    filter1(x, y, z, c) = filter(x, y, z, c);
                    filter2(x, y, z, c) = filter(x, y, z, c);
                }
            }
        }
    }

    for (int x = 0; x < bias.width(); x++) {
        bias(x) = rand();
        bias1(x) = bias(x);
        bias2(x) = bias(x);
    }

#ifdef _WIN32
    _putenv_s("HL_CUDA_JIT_MAX_REGISTERS", "256");
#else
    setenv("HL_CUDA_JIT_MAX_REGISTERS", "256", 1);
#endif
}

static void Halide_ConvLayer_NonSchedule(benchmark::State &state) {
    for (auto _ : state) {
        for (int i = 0; i < state.range(0); ++i) {
            conv_layer_nonschedule(input, filter, bias, output);
        }
    }
}

static void Halide_ConvLayer_MaunallySchedule(benchmark::State &state) {
    for (auto _ : state) {
        for (int i = 0; i < state.range(0); ++i) {
            conv_layer_manuallyschedule(input1, filter1, bias1, output1);
        }
    }
}

static void Halide_ConvLayer_AutoSchedule(benchmark::State &state) {
    for (auto _ : state) {
        for (int i = 0; i < state.range(0); ++i) {
            conv_layer_autoschedule(input2, filter2, bias2, output2);
        }
    }
}

// Register benchmarking function.
void registerBenchmarkHalideConvLayer() {
    BENCHMARK(Halide_ConvLayer_NonSchedule)->Arg(1)->Unit(benchmark::kMillisecond);
    BENCHMARK(Halide_ConvLayer_MaunallySchedule)->Arg(1)->Unit(benchmark::kMillisecond);
    BENCHMARK(Halide_ConvLayer_AutoSchedule)->Arg(1)->Unit(benchmark::kMillisecond);
}

