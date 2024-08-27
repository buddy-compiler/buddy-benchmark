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
// This file implements the benchmark for FFN layer.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

// Define target layout.
#define INPUT_DIM 256
#define HIDDEN_DIM 64
#define OUTPUT_DIM 64
#define BATCH_SIZE 1
constexpr size_t ParamsSize = 20608;

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
// Declare the FFN layer C interface.
extern "C" {
void _mlir_ciface_forward_scalar(MemRef<float, 2> *output,
                                 MemRef<float, 1> *input1,
                                 MemRef<float, 2> *input2);
void _mlir_ciface_forward_auto_vectorization(MemRef<float, 2> *output,
                                             MemRef<float, 1> *input1,
                                             MemRef<float, 2> *input2);
}

} // namespace

template <typename Func> void DL_LAYER_FFN(benchmark::State &state, Func func) {

  // Define the sizes of the input and output tensors.
  intptr_t sizesInput[2] = {BATCH_SIZE, INPUT_DIM};
  intptr_t sizesOutput[2] = {BATCH_SIZE, OUTPUT_DIM};
  intptr_t sizesParams[1] = {ParamsSize};

  MemRef<float, 2> input1(sizesInput, 2);
  MemRef<float, 2> output(sizesOutput, 0);
  MemRef<float, 1> paramsContainer(sizesParams, 3);

  for (auto _ : state) {
    func(&output, &paramsContainer, &input1);
  }
}

BENCHMARK_CAPTURE(DL_LAYER_FFN, Scalar, _mlir_ciface_forward_scalar)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_LAYER_FFN, Auto_Vectorization,
                  _mlir_ciface_forward_auto_vectorization)
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

  // Set the layout sizes of input and output memref container.
  intptr_t sizesInput[2] = {BATCH_SIZE, INPUT_DIM};
  intptr_t sizesOutput[2] = {BATCH_SIZE, OUTPUT_DIM};
  intptr_t sizesParams[1] = {ParamsSize};

  // Generate input memref containers with random numbers.
  const int inputSize = BATCH_SIZE * INPUT_DIM;
  float inputRand1[inputSize];
  float inputRand2[ParamsSize];

  for (int i = 0; i < inputSize; ++i) {
    inputRand1[i] = distribution(generator);
  }
  for (int i = 0; i < ParamsSize; ++i) {
    inputRand2[i] = distribution(generator);
  }

  MemRef<float, 2> inputMemRef(inputRand1, sizesInput);
  MemRef<float, 1> paramsContainer(inputRand2, sizesParams);

  // Generate output memref containers with zero.
  MemRef<float, 2> outputScalar(sizesOutput);
  MemRef<float, 2> outputAutoVectorization(sizesOutput);

  // Perform all the addf implementations.
  _mlir_ciface_forward_scalar(&outputScalar, &paramsContainer, &inputMemRef);
  _mlir_ciface_forward_auto_vectorization(&outputAutoVectorization,
                                          &paramsContainer, &inputMemRef);
  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultAutoVectorization = outputAutoVectorization.getData();

  // Print the verification result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification: "
            << (areArraysEqual(resultScalar, resultAutoVectorization,
                               sizesOutput[0] * sizesOutput[1])
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
