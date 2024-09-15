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
#include <iostream>
#include <random>

// Define target layout.
constexpr size_t INPUT0_DIM = 109486854; // Assuming large input size for demonstration
constexpr size_t INPUT1_DIM = 512;
constexpr size_t INPUT2_DIM = 8;  // Assuming auxiliary inputs remain multidimensional
constexpr size_t OUTPUT_DIM = 6;  // Assuming output has batch size of 1

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

extern "C" {
void _mlir_ciface_forward_auto_vectorization(MemRef<float, 2> *output,
                                      MemRef<float, 1> *input0,
                                      MemRef<int64_t, 1> *input1,
                                      MemRef<int64_t, 2> *input2,
                                      MemRef<int64_t, 2> *input3,
                                      MemRef<int64_t, 2> *input4);
void _mlir_ciface_forward_buddy_vectorization(MemRef<float, 2> *output,
                                          MemRef<float, 1> *input0,
                                          MemRef<int64_t, 1> *input1,
                                          MemRef<int64_t, 2> *input2,
                                          MemRef<int64_t, 2> *input3,
                                          MemRef<int64_t, 2> *input4);
}

template <typename Func>
void DL_LAYER_BERT(benchmark::State &state, Func func) {
  intptr_t sizesInput0[1] = {INPUT0_DIM};
  intptr_t sizesInput1[1] = {INPUT1_DIM};
  intptr_t sizesInput2[2] = {1, INPUT2_DIM};
  intptr_t sizesOutput[2] = {1, OUTPUT_DIM};

  MemRef<float, 1> input0(sizesInput0);
  MemRef<int64_t, 1> input1(sizesInput1);
  MemRef<int64_t, 2> input2(sizesInput2);
  MemRef<int64_t, 2> input3(sizesInput2);
  MemRef<int64_t, 2> input4(sizesInput2);
  MemRef<float, 2> output(sizesOutput);

  for (auto _ : state) {
    func(&output, &input0, &input1, &input2, &input3, &input4);
  }
}

BENCHMARK_CAPTURE(DL_LAYER_BERT, Auto_Vectorization, _mlir_ciface_forward_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_LAYER_BERT, Buddy_Vectorization, _mlir_ciface_forward_buddy_vectorization)
    ->Unit(benchmark::kMillisecond);

/// Correctness Verification
void verification() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  std::uniform_int_distribution<int64_t> int_distribution(0, 100);

  // Generate random inputs for float and int64 types.
  float input0Rand[INPUT0_DIM];
  int64_t input1Rand[INPUT1_DIM];
  int64_t input2Rand[INPUT2_DIM];
  int64_t input3Rand[INPUT2_DIM];
  int64_t input4Rand[INPUT2_DIM];

  for (int i = 0; i < INPUT0_DIM; ++i) {
    input0Rand[i] = distribution(generator);
  }
  for (int i = 0; i < INPUT1_DIM; ++i) {
    input1Rand[i] = int_distribution(generator);
  }
  for (int i = 0; i < INPUT2_DIM; ++i) {
    input2Rand[i] = int_distribution(generator);
    input3Rand[i] = int_distribution(generator);
    input4Rand[i] = int_distribution(generator);
  }

  // Create MemRef containers for inputs and outputs
  intptr_t sizesInput0[1] = {INPUT0_DIM};
  intptr_t sizesInput1[1] = {INPUT1_DIM};
  intptr_t sizesInput2[2] = {1, INPUT2_DIM};
  intptr_t sizesOutput[2] = {1, OUTPUT_DIM};

  MemRef<float, 1> input0MemRef(input0Rand, sizesInput0);
  MemRef<int64_t, 1> input1MemRef(input1Rand, sizesInput1);
  MemRef<int64_t, 2> input2MemRef(input2Rand, sizesInput2);
  MemRef<int64_t, 2> input3MemRef(input2Rand, sizesInput2);
  MemRef<int64_t, 2> input4MemRef(input2Rand, sizesInput2);
  MemRef<float, 2> output1(sizesOutput);
  MemRef<float, 2> output2(sizesOutput);

  // Perform scalar and vectorized operations.
  _mlir_ciface_forward_auto_vectorization(&output1, &input0MemRef, &input1MemRef, &input2MemRef, &input3MemRef, &input4MemRef);
  _mlir_ciface_forward_buddy_vectorization(&output2, &input0MemRef, &input1MemRef, &input2MemRef, &input3MemRef, &input4MemRef);

  // Compare the results.
  bool areEqual = areArraysEqual(output1.getData(), output2.getData(), OUTPUT_DIM * 1); // Multiplying by batch size

  // Print the verification result.
  std::cout << "-----------------------------------------------------------" << std::endl;
  std::cout << "Correctness Verification: " << (areEqual ? PASS : FAIL) << std::endl;
  std::cout << "-----------------------------------------------------------" << std::endl;
}


int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  verification();
  return 0;
}
