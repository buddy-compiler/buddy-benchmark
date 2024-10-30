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
// This file implements the benchmark for BERT model.
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
constexpr size_t ParamsSize = 109486854;
constexpr size_t InputSize = 512;
constexpr size_t MaxTokenLength = 8;
constexpr size_t OutputSize = 6;
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

// Declare the BERT C interface.
extern "C" {
void _mlir_ciface_forward_auto_vectorization(MemRef<float, 2> *output,
                                             MemRef<float, 1> *input0,
                                             MemRef<size_t, 1> *input1,
                                             MemRef<size_t, 2> *input2,
                                             MemRef<size_t, 2> *input3,
                                             MemRef<size_t, 2> *input4);

void _mlir_ciface_forward_buddy_vectorization(MemRef<float, 2> *output,
                                              MemRef<float, 1> *input0,
                                              MemRef<size_t, 1> *input1,
                                              MemRef<size_t, 2> *input2,
                                              MemRef<size_t, 2> *input3,
                                              MemRef<size_t, 2> *input4);
}

template <typename Func>
void DL_MODEL_BERT(benchmark::State &state, Func func) {
  MemRef<float, 2> output({1, OutputSize}, 1);
  MemRef<float, 1> input0({ParamsSize}, 2);
  MemRef<size_t, 1> input1({InputSize}, 3);
  MemRef<size_t, 2> input2({1, MaxTokenLength}, 4);
  MemRef<size_t, 2> input3({1, MaxTokenLength}, 5);
  MemRef<size_t, 2> input4({1, MaxTokenLength}, 6);

  for (auto _ : state) {
    func(&output, &input0, &input1, &input2, &input3, &input4);
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK_CAPTURE(DL_MODEL_BERT, Auto_Vectorization,
                  _mlir_ciface_forward_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_MODEL_BERT, Buddy_Vectorization,
                  _mlir_ciface_forward_buddy_vectorization)
    ->Unit(benchmark::kMillisecond);

/// Correctness Verification
void verification() {
  std::random_device rd;

  MemRef<float, 2> outputAutoVectorization({1, OutputSize}, 1);
  MemRef<float, 2> outputBuddyVectorization({1, OutputSize}, 1);
  MemRef<float, 1> input0({ParamsSize}, 2);
  MemRef<size_t, 1> input1({InputSize}, 3);
  MemRef<size_t, 2> input2({1, MaxTokenLength}, 4);
  MemRef<size_t, 2> input3({1, MaxTokenLength}, 5);
  MemRef<size_t, 2> input4({1, MaxTokenLength}, 6);

  // Call the forward functions of the model.
  _mlir_ciface_forward_auto_vectorization(&outputAutoVectorization, &input0, &input1, &input2, &input3, &input4);
  _mlir_ciface_forward_buddy_vectorization(&outputBuddyVectorization, &input0, &input1, &input2, &input3, &input4);

  auto resultAutoVectorization = outputAutoVectorization.getData();
  auto resultBuddyVectorization = outputBuddyVectorization.getData();
  size_t resultSize = OutputSize;

  // Print the verification result.
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
