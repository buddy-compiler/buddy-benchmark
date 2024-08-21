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
// This file implements the benchmark for Mobilenet-V3 model.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#define INPUT_N 1
#define INPUT_C 3
#define INPUT_H 224
#define INPUT_W 224
#define OUTPUT_N 1000

// Helper functions and variables.
namespace {
constexpr size_t ParamsSize = 1105815552;
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 2048;
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

// constexpr size_t ParamsSize = 2554968;

// bool areArraysEqual(float array1[], float array2[], int size,
//                     float epsilon = 0.0001) {
//   for (int i = 0; i < size; ++i) {
//     if (fabs(array1[i] - array2[i]) > epsilon) {
//       return false;
//     }
//   }
//   return true;
// }
} // namespace

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_forward_auto_vectorization(MemRef<float, 3> *a,
                                             MemRef<float, 1> *b,
                                             MemRef<size_t, 2> *c);

void _mlir_ciface_forward_vectorization(MemRef<float, 3> *a,
                                        MemRef<float, 1> *b,
                                        MemRef<size_t, 2> *c);
}

template <typename Func>
void BM_TinyLlama_V3(benchmark::State &state, Func func) {

  // MemRef<size_t, 2> outputContainer;
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, false, 2),
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, false, 3)};
  MemRef<size_t, 2> inputContainer({1, MaxTokenLength}, 5);
  MemRef<float, 1> paramsContainerf32({ParamsSize}, 4);

  for (auto _ : state) {
    func(resultContainer, &paramsContainerf32, &inputContainer);
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK_CAPTURE(BM_TinyLlama_V3, BM_TinyLlama_V3_Auto_Vectorization,
                  _mlir_ciface_forward_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_TinyLlama_V3, BM_TinyLlama_V3_Vectorization,
                  _mlir_ciface_forward_vectorization)
    ->Unit(benchmark::kMillisecond);

// /// Correctness Verification
// /// The verification does not affect the performance.
// /// - Set the scalar case as the criteria.
// /// - Input elements are random numbers.
// /// - Output elements are initialized to zero.
// /// - Compare the output of various optimizations with the scalar version to
// ///   verify correctness.
// void verification() {
//   // Set the random number generator.
//   std::random_device rd;
//   std::mt19937 generator(rd());
//   std::uniform_real_distribution<float> distribution(0.0, 1.0);

//   MemRef<size_t, 2> outputContainer;
//   MemRef<float, 3> resultContainer[2] = {
//       MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, false, 2),
//       MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, false, 3)};
//   MemRef<size_t, 2> inputContainer({1, MaxTokenLength}, 5);
//   MemRef<float, 1> paramsContainerf32({ParamsSize}, 4);

//   // Call the forward function of the model.
//   _mlir_ciface_forward_auto_vectorization(resultContainer,
//   &paramsContainerf32,
//                                           &inputContainer);
//   _mlir_ciface_forward_vectorization(resultContainer, &paramsContainerf32,
//                                      &inputContainer);

//   // auto resultScalar = outputScalar.getData();
//   // auto resultVectorization = outputVectorization.getData();

//   // Print the verfication result.
//   std::cout << "-----------------------------------------------------------"
//             << std::endl;
//   std::cout << "Correctness Verification:" << std::endl;
//   std::cout << "Transform case: "
//             // << (areArraysEqual(resultScalar, resultVectorization,
//             OUTPUT_N)
//             //         ? PASS
//             //         : FAIL)
//             << std::endl;
//   std::cout << "-----------------------------------------------------------"
//             << std::endl;
// }

int main(int argc, char **argv) {
  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  // Run correctness verification.
  // verification();
  return 0;
}
