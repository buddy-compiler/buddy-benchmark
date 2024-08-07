//===- GoogleBenchmarkMain.cpp---------------------------------------------===//
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
// This file implements the benchmark for softmaxExpSumDiv operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

// Define target layout.
#define BATCH_SIZE 1
#define SEQ_LENGTH 32
#define FEATURE_DIM 64

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
// Declare the softmaxExpSumDiv C interface.
extern "C" {
void _mlir_ciface_softmaxexpsumdiv_scalar(MemRef<float, 3> *input,
                                          MemRef<float, 3> *output);
void _mlir_ciface_softmaxexpsumdiv_auto_vectorization(MemRef<float, 3> *input,
                                                      MemRef<float, 3> *output);
}

#define DEFINE_SOFTMAXEXPSUMDIV_BENCHMARK(name, func)                          \
  void BM_SOFTMAXEXPSUMDIV_##name(benchmark::State &state) {                   \
    intptr_t sizesInput[3] = {BATCH_SIZE, SEQ_LENGTH, FEATURE_DIM};            \
    intptr_t sizesOutput[3] = {BATCH_SIZE, SEQ_LENGTH, FEATURE_DIM};           \
                                                                               \
    MemRef<float, 3> input(sizesInput, 1.0);                                   \
    MemRef<float, 3> output(sizesOutput, 0.0);                                 \
                                                                               \
    for (auto _ : state) {                                                     \
      func(&input, &output);                                                   \
    }                                                                          \
  }

DEFINE_SOFTMAXEXPSUMDIV_BENCHMARK(SCALAR, _mlir_ciface_softmaxexpsumdiv_scalar)
DEFINE_SOFTMAXEXPSUMDIV_BENCHMARK(
    AutoVectorization, _mlir_ciface_softmaxexpsumdiv_auto_vectorization)
} // namespace

// Register benchmark cases.
BENCHMARK(BM_SOFTMAXEXPSUMDIV_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SOFTMAXEXPSUMDIV_AutoVectorization)->Unit(benchmark::kMillisecond);

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

  // Set the layout sizes of input and output memref container.
  intptr_t sizesInput[3] = {BATCH_SIZE, SEQ_LENGTH, FEATURE_DIM};
  intptr_t sizesOutput[3] = {BATCH_SIZE, SEQ_LENGTH, FEATURE_DIM};

  // Generate input memref container with random numbers.
  const int inputSize = BATCH_SIZE * SEQ_LENGTH * FEATURE_DIM;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }
  MemRef<float, 3> inputMemRef(inputRand, sizesInput);

  // Generate output memref container with zero.
  const int outputSize = BATCH_SIZE * SEQ_LENGTH * FEATURE_DIM;
  MemRef<float, 3> outputScalar(sizesOutput, 0.0);
  MemRef<float, 3> outputAutoVectorization(sizesOutput, 0.0);

  // Perform all the softmaxExpSumDiv implementation.
  _mlir_ciface_softmaxexpsumdiv_scalar(&inputMemRef, &outputScalar);
  _mlir_ciface_softmaxexpsumdiv_auto_vectorization(&inputMemRef, &outputAutoVectorization);

  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultAutoVectorization = outputAutoVectorization.getData();

  // Print the verification result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultAutoVectorization,
                               outputSize)
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
