//===- Conv2DNchwFchwBenchmark.cpp ----------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the benchmark for GEMM operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

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

// Helper functions and variables.
namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(float array1[], float array2[], int size) {
  for (int i = 0; i < size; ++i) {
    if (array1[i] != array2[i]) {
      return false;
    }
  }
  return true;
}
} // namespace

namespace {
// Declare the C interface.
extern "C" {
void _mlir_ciface_conv2d_nchw_fchw_scalar(MemRef<float, 4> *input,
                                          MemRef<float, 4> *filter,
                                          MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_ocv(MemRef<float, 4> *input,
                                       MemRef<float, 4> *filter,
                                       MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_winagrad(MemRef<float, 4> *input,
                                            MemRef<float, 4> *filter,
                                            MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_tiling(MemRef<float, 4> *input,
                                                 MemRef<float, 4> *filter,
                                                 MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_16(MemRef<float, 4> *input,
                                             MemRef<float, 4> *filter,
                                             MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_32(MemRef<float, 4> *input,
                                             MemRef<float, 4> *filter,
                                             MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_64(MemRef<float, 4> *input,
                                             MemRef<float, 4> *filter,
                                             MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_128(MemRef<float, 4> *input,
                                              MemRef<float, 4> *filter,
                                              MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col_256(MemRef<float, 4> *input,
                                              MemRef<float, 4> *filter,
                                              MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_broadcast_16(MemRef<float, 4> *input,
                                                MemRef<float, 4> *filter,
                                                MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_broadcast_32(MemRef<float, 4> *input,
                                                MemRef<float, 4> *filter,
                                                MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_broadcast_64(MemRef<float, 4> *input,
                                                MemRef<float, 4> *filter,
                                                MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_broadcast_128(MemRef<float, 4> *input,
                                                 MemRef<float, 4> *filter,
                                                 MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_broadcast_256(MemRef<float, 4> *input,
                                                 MemRef<float, 4> *filter,
                                                 MemRef<float, 4> *output);
}

#define DEFINE_BENCHMARK(name, func)                                           \
  void BM_CONV2D_NCHW_FCHW_##name(benchmark::State &state) {                   \
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

DEFINE_BENCHMARK(SCALAR, _mlir_ciface_conv2d_nchw_fchw_scalar)
DEFINE_BENCHMARK(OCV, _mlir_ciface_conv2d_nchw_fchw_ocv)
DEFINE_BENCHMARK(WINAGRAD, _mlir_ciface_conv2d_nchw_fchw_winagrad)
DEFINE_BENCHMARK(IM2COL_TILING, _mlir_ciface_conv2d_nchw_fchw_im2col_tiling)
DEFINE_BENCHMARK(IM2COL_16, _mlir_ciface_conv2d_nchw_fchw_im2col_16)
DEFINE_BENCHMARK(IM2COL_32, _mlir_ciface_conv2d_nchw_fchw_im2col_32)
DEFINE_BENCHMARK(IM2COL_64, _mlir_ciface_conv2d_nchw_fchw_im2col_64)
DEFINE_BENCHMARK(IM2COL_128, _mlir_ciface_conv2d_nchw_fchw_im2col_128)
DEFINE_BENCHMARK(IM2COL_256, _mlir_ciface_conv2d_nchw_fchw_im2col_256)
DEFINE_BENCHMARK(BROADCAST_16, _mlir_ciface_conv2d_nchw_fchw_broadcast_16)
DEFINE_BENCHMARK(BROADCAST_32, _mlir_ciface_conv2d_nchw_fchw_broadcast_32)
DEFINE_BENCHMARK(BROADCAST_64, _mlir_ciface_conv2d_nchw_fchw_broadcast_64)
DEFINE_BENCHMARK(BROADCAST_128, _mlir_ciface_conv2d_nchw_fchw_broadcast_128)
DEFINE_BENCHMARK(BROADCAST_256, _mlir_ciface_conv2d_nchw_fchw_broadcast_256)
} // namespace

BENCHMARK(BM_CONV2D_NCHW_FCHW_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_OCV)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_WINAGRAD)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_IM2COL_TILING)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_IM2COL_16)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_IM2COL_32)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_IM2COL_64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_IM2COL_128)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_IM2COL_256)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_BROADCAST_16)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_BROADCAST_32)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_BROADCAST_64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_BROADCAST_128)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_BROADCAST_256)->Unit(benchmark::kMillisecond);

#define DEFINE_VERIFICATION(name, func)                                        \
  void VERIFICATION_##name(MemRef<float, 4> inputMemRef,                       \
                           MemRef<float, 4> kernelMemRef,                      \
                           float resultScalar[]) {                             \
    intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};        \
    MemRef<float, 4> output##name(sizesOutput, 0);                             \
    func(&inputMemRef, &kernelMemRef, &output##name);                          \
    auto result##name = output##name.getData();                                \
    const int outputSize = OUTPUT_N * OUTPUT_F * OUTPUT_H * OUTPUT_W;          \
    std::cout << #name << " case: "                                            \
              << (areArraysEqual(resultScalar, result##name, outputSize)       \
                      ? PASS                                                   \
                      : FAIL)                                                  \
              << std::endl;                                                    \
  }

DEFINE_VERIFICATION(OCV, _mlir_ciface_conv2d_nchw_fchw_ocv)
DEFINE_VERIFICATION(WINAGRAD, _mlir_ciface_conv2d_nchw_fchw_winagrad)
DEFINE_VERIFICATION(IM2COL_TILING, _mlir_ciface_conv2d_nchw_fchw_im2col_tiling)
DEFINE_VERIFICATION(IM2COL_16, _mlir_ciface_conv2d_nchw_fchw_im2col_16)
DEFINE_VERIFICATION(IM2COL_32, _mlir_ciface_conv2d_nchw_fchw_im2col_32)
DEFINE_VERIFICATION(IM2COL_64, _mlir_ciface_conv2d_nchw_fchw_im2col_64)
DEFINE_VERIFICATION(IM2COL_128, _mlir_ciface_conv2d_nchw_fchw_im2col_128)
DEFINE_VERIFICATION(IM2COL_256, _mlir_ciface_conv2d_nchw_fchw_im2col_256)
DEFINE_VERIFICATION(BROADCAST_16, _mlir_ciface_conv2d_nchw_fchw_broadcast_16)
DEFINE_VERIFICATION(BROADCAST_32, _mlir_ciface_conv2d_nchw_fchw_broadcast_32)
DEFINE_VERIFICATION(BROADCAST_64, _mlir_ciface_conv2d_nchw_fchw_broadcast_64)
DEFINE_VERIFICATION(BROADCAST_128, _mlir_ciface_conv2d_nchw_fchw_broadcast_128)
DEFINE_VERIFICATION(BROADCAST_256, _mlir_ciface_conv2d_nchw_fchw_broadcast_256)

void verification() {
  // Set the random number generator.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(1, 100);

  // Set the layout sizes of input and output memref container.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};

  // Generate input memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }
  MemRef<float, 4> inputMemRef(inputRand, sizesInput);

  // Generate kernel memref container with random numbers.
  const int kernelSize = KERNEL_F * KERNEL_C * KERNEL_H * KERNEL_W;
  float kernelRand[kernelSize];
  for (int i = 0; i < kernelSize; ++i) {
    kernelRand[i] = distribution(generator);
  }
  MemRef<float, 4> kernelMemRef(kernelRand, sizesKernel);

  // Generate a result using a scalar method for comparison during verification.
  MemRef<float, 4> outputScalar(sizesOutput, 0);
  _mlir_ciface_conv2d_nchw_fchw_scalar(&inputMemRef, &kernelMemRef,
                                       &outputScalar);
  auto resultScalar = outputScalar.getData();

  // Print the verfication result.
  std::cout << "---------------------------------------------------------------"
               "---------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;

  VERIFICATION_OCV(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_WINAGRAD(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_IM2COL_TILING(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_IM2COL_16(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_IM2COL_32(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_IM2COL_64(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_IM2COL_128(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_IM2COL_256(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_BROADCAST_16(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_BROADCAST_32(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_BROADCAST_64(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_BROADCAST_128(inputMemRef, kernelMemRef, resultScalar);
  VERIFICATION_BROADCAST_256(inputMemRef, kernelMemRef, resultScalar);

  std::cout << "---------------------------------------------------------------"
               "---------"
            << std::endl;
}
