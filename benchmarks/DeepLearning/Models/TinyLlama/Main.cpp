//===- Main.cpp -----------------------------------------------------------===//
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

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

constexpr size_t ParamsSize = 1105815552;
constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t HiddenSize = 2048;

/// TODO: Initialize input data with random numbers.
MemRef<size_t, 2> inputContainer({1, MaxTokenLength}, 4);
MemRef<float, 1> paramsContainerf32({ParamsSize}, 5);

// Runs the MLIR-based function for benchmarking.
template <typename Func>
void DL_MODEL_TINYLLAMA(benchmark::State &state, Func func) {
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, 2),
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, 3)};
  for (auto _ : state) {
    func(resultContainer, &paramsContainerf32, &inputContainer);
  }
}

using MLIRFunctionType = void (*)(MemRef<float, 3> *, MemRef<float, 1> *,
                                  MemRef<size_t, 2> *);
//  Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(float *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, 2),
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, 3)};
  MLIRFunc(resultContainer, &paramsContainerf32, &inputContainer);
  float *outputOptimized = resultContainer[0].getData();
  size_t size = resultContainer[0].getSize();
  tinyllama::verify<float>(outputExpected, outputOptimized, size, name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_forward_scalar(MemRef<float, 3> *a, MemRef<float, 1> *b,
                                 MemRef<size_t, 2> *c);
void _mlir_ciface_forward_matmul_opt(MemRef<float, 3> *a, MemRef<float, 1> *b,
                                     MemRef<size_t, 2> *c);
/// [Step 1] Add function of your new method.
}
BENCHMARK_CAPTURE(DL_MODEL_TINYLLAMA, scalar, _mlir_ciface_forward_scalar)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_MODEL_TINYLLAMA, matmul_opt,
                  _mlir_ciface_forward_matmul_opt)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Attain scalar output results as expected output results in verification.
  MemRef<float, 3> resultContainer[2] = {
      MemRef<float, 3>({1, MaxTokenLength, HiddenSize}, 2),
      MemRef<float, 3>({1, MaxTokenLength, MaxVocabSize}, 3)};
  _mlir_ciface_forward_scalar(resultContainer, &paramsContainerf32,
                              &inputContainer);
  float *outputExpected = resultContainer[0].getData();

  MLIRVerification(outputExpected, _mlir_ciface_forward_matmul_opt,
                   "matmul_opt");
  /// [Step 3] Add your new method for verification.
  return 0;
}
