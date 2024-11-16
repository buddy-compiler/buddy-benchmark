//===- GoogleBenchmarkMain.cpp --------------------------------------------===//
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
// This file implements the benchmark for conv2d(nchw-fchw) operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#define INPUT_N 1
#define INPUT_C 10
#define INPUT_H 28
#define INPUT_W 28
#define KERNEL_H 5
#define KERNEL_W 5
#define OUTPUT_N 1
#define OUTPUT_C 10
#define OUTPUT_H 12
#define OUTPUT_W 12

// Helper functions and variables.
namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(float array1[], float array2[], int size,
                    float epsilon = 0.0001) {
  for (int i = 0; i < size; ++i) {
    if (fabs(array1[i] - array2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

} // namespace

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_pooling_nhwc_max_scalar(MemRef<float, 4> *input,
                                           MemRef<float, 2> *filter,
                                           MemRef<float, 4> *output);

void _mlir_ciface_pooling_nhwc_max_vec(MemRef<float, 4> *input,
                                           MemRef<float, 2> *filter,
                                           MemRef<float, 4> *output);
void _mlir_ciface_pooling_nhwc_max_adapt_vec(MemRef<float, 4> *input,
                                           MemRef<float, 2> *filter,
                                           MemRef<float, 4> *output);
}

#define DEFINE_PoolingNhwcMax_BENCHMARK(name, func)                            \
  void BM_PoolingNhwcMax_##name(benchmark::State &state) {                     \
    intptr_t sizesInput[4] = {INPUT_N, INPUT_H, INPUT_W, INPUT_C};             \
    intptr_t sizesKernel[2] = {KERNEL_H, KERNEL_W};        \
    intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_H, OUTPUT_W, OUTPUT_C};        \
                                                                               \
    MemRef<float, 4> inputMemRef(sizesInput, 2.0);                             \
    MemRef<float, 2> filterMemRef(sizesKernel, 3.0);                           \
    MemRef<float, 4> outputMemRef(sizesOutput, 0.0);                           \
                                                                               \
    for (auto _ : state) {                                                     \
      func(&inputMemRef, &filterMemRef, &outputMemRef);                        \
    }                                                                          \
  }

DEFINE_PoolingNhwcMax_BENCHMARK(SCALAR, _mlir_ciface_pooling_nhwc_max_scalar)
DEFINE_PoolingNhwcMax_BENCHMARK(Vectorization,
                                _mlir_ciface_pooling_nhwc_max_vec)
DEFINE_PoolingNhwcMax_BENCHMARK(Adapt_Vectorization,
                                _mlir_ciface_pooling_nhwc_max_adapt_vec)

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_PoolingNhwcMax_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_PoolingNhwcMax_Vectorization)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_PoolingNhwcMax_Adapt_Vectorization)->Unit(benchmark::kMillisecond);

/// Correctness Verification
/// The verification does not affect the performance.
/// - Set the scalar case as the criteria.
/// - Input elements are random numbers.
/// - Output elements are initialized to zero.
/// - Compare the output of various optimizations with the scalar version to
///   verify correctness.
void verification() {
  // Set the random number generator.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  // Set the layout sizes of input and kernel memref container.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_H, INPUT_W, INPUT_C};
  intptr_t sizesKernel[2] = {KERNEL_H, KERNEL_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_H, OUTPUT_W, OUTPUT_C};

  // Generate input memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }
  MemRef<float, 4> inputMemRef(inputRand, sizesInput);

  // Generate kernel memref container with random numbers.
  const int kernelSize = KERNEL_H * KERNEL_W;
  float kernelRand[kernelSize];
  for (int i = 0; i < kernelSize; ++i) {
    kernelRand[i] = distribution(generator);
  }
  MemRef<float, 2> filterMemRef(kernelRand, sizesKernel);

  // Generate output memref container with zero.
  MemRef<float, 4> outputMemRef(sizesOutput, 0.0);
  MemRef<float, 4> outputTransform(sizesOutput, 0.0);
  MemRef<float, 4> outputAdaptTransform(sizesOutput, 0.0);

  // Perform all the matmul implementation.
  _mlir_ciface_pooling_nhwc_max_scalar(&inputMemRef, &filterMemRef,
                                        &outputMemRef);
  _mlir_ciface_pooling_nhwc_max_vec(&inputMemRef, &filterMemRef,
                                        &outputTransform);
  _mlir_ciface_pooling_nhwc_max_adapt_vec(&inputMemRef, &filterMemRef,
                                        &outputAdaptTransform);

  // Get the result array.
  auto resultScalar = outputMemRef.getData();
  auto resultTransform = outputTransform.getData();
  auto resultAdaptTransform = outputAdaptTransform.getData();
  

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultTransform,
                               OUTPUT_N * OUTPUT_C * OUTPUT_H * OUTPUT_W)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultAdaptTransform,
                               OUTPUT_N * OUTPUT_C * OUTPUT_H * OUTPUT_W)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}

int main(int argc, char **argv) {
  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  // Run correctness verification.
  verification();
  return 0;
}
