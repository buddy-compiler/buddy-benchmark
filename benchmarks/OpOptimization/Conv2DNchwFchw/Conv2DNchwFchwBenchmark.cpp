//===- MatmulOpsBenchmark.cpp
//------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

namespace {
extern "C" {

void _mlir_ciface_conv2d_nchw_fchw(MemRef<float, 4> *input,
                                   MemRef<float, 4> *filter,
                                   MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_vector_8(MemRef<float, 4> *input,
                                            MemRef<float, 4> *filter,
                                            MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_vector_16(MemRef<float, 4> *input,
                                             MemRef<float, 4> *filter,
                                             MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_vector_32(MemRef<float, 4> *input,
                                             MemRef<float, 4> *filter,
                                             MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_vector_64(MemRef<float, 4> *input,
                                             MemRef<float, 4> *filter,
                                             MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_vector_128(MemRef<float, 4> *input,
                                              MemRef<float, 4> *filter,
                                              MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_vector_8(MemRef<float, 4> *input,
                                                   MemRef<float, 4> *filter,
                                                   MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_vector_16(MemRef<float, 4> *input,
                                                    MemRef<float, 4> *filter,
                                                    MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_vector_32(MemRef<float, 4> *input,
                                                    MemRef<float, 4> *filter,
                                                    MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_vector_64(MemRef<float, 4> *input,
                                                    MemRef<float, 4> *filter,
                                                    MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_vector_128(MemRef<float, 4> *input,
                                                     MemRef<float, 4> *filter,
                                                     MemRef<float, 4> *output);
}

// Define target layout.
#define INPUT_N 1
#define INPUT_C 64
#define INPUT_H 58
#define INPUT_W 58
#define KERNEL_F 64
#define KERNEL_C 64
#define KERNEL_H 3
#define KERNEL_W 3
#define OUTPUT_N 1
#define OUTPUT_F 64
#define OUTPUT_H 56
#define OUTPUT_W 56

#define DEFINE_BENCHMARK(name, func)                                           \
  void name(benchmark::State &state) {                                         \
    intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};             \
    intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};        \
    intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};        \
    MemRef<float, 4> input(sizesInput, 1.0);                                   \
    MemRef<float, 4> filter(sizesKernel, 1.0);                                 \
    MemRef<float, 4> output(sizesOutput, 0);                                   \
    for (auto _ : state) {                                                     \
      func(&input, &filter, &output);                                          \
    }                                                                          \
  }

#define RUN_BENCHMARK(name, func)                                              \
  DEFINE_BENCHMARK(name, func) BENCHMARK(name)->Unit(benchmark::kMillisecond); \
  DEFINE_BENCHMARK(name##Vector8, func##_vector_8)                             \
  BENCHMARK(name##Vector8)->Unit(benchmark::kMillisecond);                     \
  DEFINE_BENCHMARK(name##Vector16, func##_vector_16)                           \
  BENCHMARK(name##Vector16)->Unit(benchmark::kMillisecond);                    \
  DEFINE_BENCHMARK(name##Vector32, func##_vector_32)                           \
  BENCHMARK(name##Vector32)->Unit(benchmark::kMillisecond);                    \
  DEFINE_BENCHMARK(name##Vector64, func##_vector_64)                           \
  BENCHMARK(name##Vector64)->Unit(benchmark::kMillisecond);                    \
  DEFINE_BENCHMARK(name##Vector128, func##_vector_128)                         \
  BENCHMARK(name##Vector128)->Unit(benchmark::kMillisecond);                   \
  DEFINE_BENCHMARK(name##Im2colVector8, func##_im2col_vector_8)                \
  BENCHMARK(name##Im2colVector8)->Unit(benchmark::kMillisecond);               \
  DEFINE_BENCHMARK(name##Im2colVector16, func##_im2col_vector_16)              \
  BENCHMARK(name##Im2colVector16)->Unit(benchmark::kMillisecond);              \
  DEFINE_BENCHMARK(name##Im2colVector32, func##_im2col_vector_32)              \
  BENCHMARK(name##Im2colVector32)->Unit(benchmark::kMillisecond);              \
  DEFINE_BENCHMARK(name##Im2colVector64, func##_im2col_vector_64)              \
  BENCHMARK(name##Im2colVector64)->Unit(benchmark::kMillisecond);              \
  DEFINE_BENCHMARK(name##Im2colVector128, func##_im2col_vector_128)            \
  BENCHMARK(name##Im2colVector128)->Unit(benchmark::kMillisecond);

RUN_BENCHMARK(Conv2DNchwFchw, _mlir_ciface_conv2d_nchw_fchw)

} // namespace
