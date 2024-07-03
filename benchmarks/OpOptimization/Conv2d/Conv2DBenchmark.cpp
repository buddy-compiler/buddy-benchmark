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
#define INPUT_R 10
#define INPUT_C 10
#define KERNEL_R 3
#define KERNEL_C 3
#define OUTPUT_R 8
#define OUTPUT_C 8

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
void _mlir_ciface_conv2d_scalar(MemRef<float, 2> *input,
                                          MemRef<float, 2> *filter,
                                          MemRef<float, 2> *output);
void _mlir_ciface_conv2d_rvv(MemRef<float, 2> *input,
                                       MemRef<float, 2> *filter,
                                       MemRef<float, 2> *output);
}

#define DEFINE_BENCHMARK(name, func)                                           \
  void BM_CONV2D_##name(benchmark::State &state) {                   \
    intptr_t sizesInput[2] = {INPUT_R, INPUT_C};           \
    intptr_t sizesKernel[2] = {KERNEL_R, KERNEL_C};        \
    intptr_t sizesOutput[2] = {OUTPUT_R, OUTPUT_C};        \
    MemRef<float, 2> input(sizesInput, 1.0);                                   \
    MemRef<float, 2> filter(sizesKernel, 1.0);                                 \
    MemRef<float, 2> output(sizesOutput, 0);                                   \
    for (auto _ : state) {                                                     \
      func(&input, &filter, &output);                                          \
    }                                                                          \
  }

DEFINE_BENCHMARK(SCALAR, _mlir_ciface_conv2d_scalar)
DEFINE_BENCHMARK(RVV, _mlir_ciface_conv2d_rvv)
} // namespace

BENCHMARK(BM_CONV2D_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_CONV2D_RVV)->Unit(benchmark::kMillisecond);


#define DEFINE_VERIFICATION(name, func)                                         \
  void VERIFICATION_##name(MemRef<float, 2> inputMemRef,                        \
                           MemRef<float, 2> kernelMemRef,                       \
                           float resultScalar[]) {                              \
    intptr_t sizesOutput[2] = {OUTPUT_R, OUTPUT_C};                             \
    MemRef<float, 2> output##name(sizesOutput, 0);                              \
    func(&inputMemRef, &kernelMemRef, &output##name);                           \
    auto result##name = output##name.getData();                                 \
    const int outputSize = OUTPUT_R * OUTPUT_C;                                 \
    std::cout << #name << " case: "                                             \
              << (areArraysEqual(resultScalar, result##name, outputSize)        \
                      ? PASS                                                    \
                      : FAIL)                                                   \
              << std::endl;                                                     \
  }


DEFINE_VERIFICATION(RVV, _mlir_ciface_conv2d_rvv)

void verification() {
  // Set the random number generator.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(1, 100);

  // Set the layout sizes of input and output memref container.
  intptr_t sizesInput[2] = {INPUT_R, INPUT_C};
  intptr_t sizesKernel[2] = {KERNEL_R, KERNEL_C};
  intptr_t sizesOutput[2] = {OUTPUT_R, OUTPUT_C};

  // Generate input memref container with random numbers.
  const int inputSize = INPUT_R * INPUT_C;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }
  MemRef<float, 2> inputMemRef(inputRand, sizesInput);

  // Generate kernel memref container with random numbers.
  const int kernelSize = KERNEL_R * KERNEL_C;
  float kernelRand[kernelSize];
  for (int i = 0; i < kernelSize; ++i) {
    kernelRand[i] = distribution(generator);
  }
  MemRef<float, 2> kernelMemRef(kernelRand, sizesKernel);

  // Generate a result using a scalar method for comparison during verification.
  MemRef<float, 2> outputScalar(sizesOutput, 0);
  _mlir_ciface_conv2d_scalar(&inputMemRef, &kernelMemRef, &outputScalar);
  auto resultScalar = outputScalar.getData();

  // Print the verfication result.
  std::cout << "---------------------------------------------------------------"
               "---------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;

  VERIFICATION_RVV(inputMemRef, kernelMemRef, resultScalar);

  std::cout << "---------------------------------------------------------------"
               "---------"
            << std::endl;
}