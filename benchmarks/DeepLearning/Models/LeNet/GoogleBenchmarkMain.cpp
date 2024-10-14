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
// This file implements the benchmark for LeNet model.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#define INPUT_N 1
#define INPUT_C 1
#define INPUT_H 28
#define INPUT_W 28
#define OUTPUT_N 10

// Helper functions and variables.
namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

constexpr size_t ParamsSize = 44426;

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

// Declare the target model C interface.
extern "C" {
void _mlir_ciface_forward_auto_vectorization(MemRef<float, 2> *output,
                                             MemRef<float, 1> *arg0,
                                             MemRef<float, 4> *input);

void _mlir_ciface_forward_buddy_vectorization(MemRef<float, 2> *output,
                                              MemRef<float, 1> *arg0,
                                              MemRef<float, 4> *input);
}

template <typename Func>
void DL_MODEL_LENET(benchmark::State &state, Func func) {

  // Define the sizes of the input and output tensors.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesOutput[2] = {1, OUTPUT_N};

  // Generate input memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;

  // Create input and output containers for the image and model output.
  MemRef<float, 4> input(sizesInput);
  MemRef<float, 2> output(sizesOutput);

  // Set random model parameters.
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 2.0);

  for (auto _ : state) {
    func(&output, &paramsContainerf32, &input);
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK_CAPTURE(DL_MODEL_LENET, Auto_Vectorization,
                  _mlir_ciface_forward_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_MODEL_LENET, Buddy_Vectorization,
                  _mlir_ciface_forward_buddy_vectorization)
    ->Unit(benchmark::kMillisecond);

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

  // Define the sizes of the input and output tensors.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesOutput[2] = {1, OUTPUT_N};

  // Generate input memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }

  // Create input and output containers for the image and model output.
  MemRef<float, 4> input(inputRand, sizesInput);
  MemRef<float, 2> outputAutoVectorization(sizesOutput, 0);
  MemRef<float, 2> outputBuddyVectorization(sizesOutput, 0);

  // Load model parameters from the specified file.
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 3.0);

  // Call the forward function of the model.
  _mlir_ciface_forward_auto_vectorization(&outputAutoVectorization,
                                          &paramsContainerf32, &input);
  _mlir_ciface_forward_buddy_vectorization(&outputBuddyVectorization,
                                           &paramsContainerf32, &input);

  auto resultAutoVectorization = outputAutoVectorization.getData();
  auto resultBuddyVectorization = outputBuddyVectorization.getData();

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultAutoVectorization,
                               resultBuddyVectorization, OUTPUT_N)
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
