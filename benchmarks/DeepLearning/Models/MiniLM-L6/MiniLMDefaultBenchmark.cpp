//===- MiniLMDefaultBenchmark.cpp -----------------------------------------===//
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
// This file implements the benchmark for e2e MiniLM-L6.
// The MiniLM-L6.mlir is generated from torch-mlir project.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <iostream>
#include <math.h>
#include <string>

namespace {

// Declare the model C interface.
extern "C" {
void _mlir_ciface_forward(MemRef<float, 2> *output,
                          buddy::Text<long long, 2> *input);
}

std::string str = "buddy compiler is a domain specific compiler!";
buddy::Text<long long, 2> input(str);
intptr_t sizesOutput[2] = {1, 2};
MemRef<float, 2> output(sizesOutput);

// Define benchmark function.
void BM_MiniLM(benchmark::State &state) {
  input.tokenize("../../benchmarks/DeepLearning/Models/MiniLM-L6/Vocab.txt",
                 200);
  for (auto _ : state) {
    _mlir_ciface_forward(&output, &input);
  }
}

// Softmax function.
void softmax(float *input, size_t size) {
  assert(size > 0);
  float m = input[0]; // Find the maximum value
  for (size_t i = 1; i < size; ++i) {
    if (input[i] > m) {
      m = input[i];
    }
  }
  float sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += expf(input[i] - m);
  }
  float constant = m + logf(sum);
  for (size_t i = 0; i < size; ++i) {
    input[i] = expf(input[i] - constant);
  }
}
} // namespace

// Register benchmarking function.
BENCHMARK(BM_MiniLM)->Unit(benchmark::kMillisecond);

// Print result function.
void printResult() {
  auto out = output.getData();
  softmax(out, 2);
  std::cout << std::string(53, '-') << std::endl;
  std::cout << "Input: " << str << std::endl;
  printf("The probability of positive label: %.2lf\n", out[1]);
  printf("The probability of negative label: %.2lf\n", out[0]);
}
