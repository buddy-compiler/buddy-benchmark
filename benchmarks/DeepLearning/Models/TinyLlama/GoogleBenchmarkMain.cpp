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
// This file implements the benchmark for Tiny LLaMA model.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// Helper functions and variables.
namespace {
constexpr size_t ParamsSize = 1105815552;
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 2048;
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
void _mlir_ciface_forward_auto_vectorization(MemRef<float, 3> *a,
                                             MemRef<float, 1> *b,
                                             MemRef<size_t, 2> *c);

void _mlir_ciface_forward_buddy_vectorization(MemRef<float, 3> *a,
                                              MemRef<float, 1> *b,
                                              MemRef<size_t, 2> *c);
}

template <typename Func>
void DL_MODEL_TinyLlama(benchmark::State &state, Func func) {

  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, 2),
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, 3)};
  MemRef<size_t, 2> inputContainer({1, MaxTokenLength}, 4);
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 5);

  for (auto _ : state) {
    func(resultContainer, &paramsContainerf32, &inputContainer);
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK_CAPTURE(DL_MODEL_TinyLlama, Auto_Vectorization,
                  _mlir_ciface_forward_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_MODEL_TinyLlama, Buddy_Vectorization,
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

  // MemRef<size_t, 2> outputContainer;
  MemRef<float, 3> resultAutoVectorizationContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, 2),
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, 3)};
  MemRef<float, 3> resultBuddyVectorizationContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, 2),
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, 3)};
  MemRef<size_t, 2> inputContainer({1, MaxTokenLength}, 4);
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 5);

  // Call the forward function of the model.
  _mlir_ciface_forward_auto_vectorization(resultAutoVectorizationContainer,
                                          &paramsContainerf32, &inputContainer);
  _mlir_ciface_forward_buddy_vectorization(
      resultBuddyVectorizationContainer, &paramsContainerf32, &inputContainer);

  auto resultAutoVectorization = resultAutoVectorizationContainer[0].getData();
  auto resultBuddyVectorization =
      resultBuddyVectorizationContainer[0].getData();
  size_t resultSize = resultAutoVectorizationContainer[0].getSize();

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification: "
            << (areArraysEqual(resultAutoVectorization,
                               resultBuddyVectorization, resultSize)
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
