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
void _mlir_ciface_conv2d_nchw_fchw_ocv(MemRef<float, 4> *input,
                                       MemRef<float, 4> *filter,
                                       MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_broadcast(MemRef<float, 4> *input,
                                             MemRef<float, 4> *filter,
                                             MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_im2col(MemRef<float, 4> *input,
                                          MemRef<float, 4> *filter,
                                          MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_winagrad(MemRef<float, 4> *input,
                                            MemRef<float, 4> *filter,
                                            MemRef<float, 4> *output);
void _mlir_ciface_conv2d_nchw_fchw_scalar(MemRef<float, 4> *input,
                                          MemRef<float, 4> *filter,
                                          MemRef<float, 4> *output);
}

void BM_CONV2D_NCHW_FCHW_OCV(benchmark::State &state) {
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};
  MemRef<float, 4> input(sizesInput, 1.0);
  MemRef<float, 4> filter(sizesKernel, 1.0);
  MemRef<float, 4> output(sizesOutput, 0);
  for (auto _ : state) {
    _mlir_ciface_conv2d_nchw_fchw_ocv(&input, &filter, &output);
  }
}

void BM_CONV2D_NCHW_FCHW_BROADCAST(benchmark::State &state) {
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};
  MemRef<float, 4> input(sizesInput, 1.0);
  MemRef<float, 4> filter(sizesKernel, 1.0);
  MemRef<float, 4> output(sizesOutput, 0);
  for (auto _ : state) {
    _mlir_ciface_conv2d_nchw_fchw_broadcast(&input, &filter, &output);
  }
}

void BM_CONV2D_NCHW_FCHW_IM2COL(benchmark::State &state) {
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};
  MemRef<float, 4> input(sizesInput, 1.0);
  MemRef<float, 4> filter(sizesKernel, 1.0);
  MemRef<float, 4> output(sizesOutput, 0);
  for (auto _ : state) {
    _mlir_ciface_conv2d_nchw_fchw_im2col(&input, &filter, &output);
  }
}

void BM_CONV2D_NCHW_FCHW_WINAGRAD(benchmark::State &state) {
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};
  MemRef<float, 4> input(sizesInput, 1.0);
  MemRef<float, 4> filter(sizesKernel, 1.0);
  MemRef<float, 4> output(sizesOutput, 0);
  for (auto _ : state) {
    _mlir_ciface_conv2d_nchw_fchw_winagrad(&input, &filter, &output);
  }
}

void BM_CONV2D_NCHW_FCHW_SCALAR(benchmark::State &state) {
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};
  MemRef<float, 4> input(sizesInput, 1.0);
  MemRef<float, 4> filter(sizesKernel, 1.0);
  MemRef<float, 4> output(sizesOutput, 0);
  for (auto _ : state) {
    _mlir_ciface_conv2d_nchw_fchw_scalar(&input, &filter, &output);
  }
}
} // namespace

BENCHMARK(BM_CONV2D_NCHW_FCHW_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_OCV)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_BROADCAST)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_IM2COL)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_NCHW_FCHW_WINAGRAD)->Unit(benchmark::kMillisecond);

void verification() {
  // Set the random number generator.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(1, 100);

  // Set the layout sizes of input and output memref container.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_F, OUTPUT_H, OUTPUT_W};

  // Generate input and kernel memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }
  MemRef<float, 4> inputMemRef(inputRand, sizesInput);

  const int kernelSize = KERNEL_F * KERNEL_C * KERNEL_H * KERNEL_W;
  float kernelRand[kernelSize];
  for (int i = 0; i < kernelSize; ++i) {
    kernelRand[i] = distribution(generator);
  }
  MemRef<float, 4> kernelMemRef(kernelRand, sizesKernel);

  // Generate output memref container with zero.
  const int outputSize = OUTPUT_N * OUTPUT_F * OUTPUT_H * OUTPUT_W;
  MemRef<float, 4> outputScalar(sizesOutput, 0);
  MemRef<float, 4> outputOCV(sizesOutput, 0);
  MemRef<float, 4> outputBroadcast(sizesOutput, 0);
  MemRef<float, 4> outputIm2col(sizesOutput, 0);
  MemRef<float, 4> outputWinagrad(sizesOutput, 0);

  // Perform all the convoluation implementation.
  _mlir_ciface_conv2d_nchw_fchw_scalar(&inputMemRef, &kernelMemRef,
                                       &outputScalar);
  _mlir_ciface_conv2d_nchw_fchw_ocv(&inputMemRef, &kernelMemRef, &outputOCV);
  _mlir_ciface_conv2d_nchw_fchw_broadcast(&inputMemRef, &kernelMemRef,
                                          &outputBroadcast);
  _mlir_ciface_conv2d_nchw_fchw_im2col(&inputMemRef, &kernelMemRef,
                                       &outputIm2col);
  _mlir_ciface_conv2d_nchw_fchw_winagrad(&inputMemRef, &kernelMemRef,
                                         &outputWinagrad);

  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultOCV = outputOCV.getData();
  auto resultBroadcast = outputBroadcast.getData();
  auto resultIm2col = outputIm2col.getData();
  auto resultWinagrad = outputWinagrad.getData();

  // Print the verfication result.
  std::cout << "---------------------------------------------------------------"
               "---------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "OCV case: "
            << (areArraysEqual(resultScalar, resultOCV, outputSize) ? PASS
                                                                    : FAIL)
            << std::endl;
  std::cout << "Broadcast case: "
            << (areArraysEqual(resultScalar, resultBroadcast, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Im2col case: "
            << (areArraysEqual(resultScalar, resultIm2col, outputSize) ? PASS
                                                                       : FAIL)
            << std::endl;
  std::cout << "Winagrad case: "
            << (areArraysEqual(resultScalar, resultWinagrad, outputSize) ? PASS
                                                                         : FAIL)
            << std::endl;
  std::cout << "---------------------------------------------------------------"
               "---------"
            << std::endl;
}
