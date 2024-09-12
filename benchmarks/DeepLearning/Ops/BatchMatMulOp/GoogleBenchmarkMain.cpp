//===----------GoogleBenchmarkMain.cpp----------===//
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
// This file implements the benchmark for Batch Matmul.
//===----------------------------------------------------------------------===//
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

// Define target layout.
#define BATCH_SIZE 3
#define M 128
#define N 128
#define K 128

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
// Declare the batch matmul C interface.
extern "C" {
void _mlir_ciface_batch_matmul_scalar(MemRef<float, 3> *input1,
                                      MemRef<float, 3> *input2,
                                      MemRef<float, 3> *output);
void _mlir_ciface_batch_matmul_auto_vectorization(MemRef<float, 3> *input1,
                                                  MemRef<float, 3> *input2,
                                                  MemRef<float, 3> *output);
void _mlir_ciface_batch_matmul_vectorization(MemRef<float, 3> *input1,
                                             MemRef<float, 3> *input2,
                                             MemRef<float, 3> *output);
void _mlir_ciface_batch_matmul_tilling_vectorization(MemRef<float, 3> *input1,
                                                     MemRef<float, 3> *input2,
                                                     MemRef<float, 3> *output);
}

template <typename Func>
void DL_OPS_BATCH_MATMUL(benchmark::State &state, Func func) {
  intptr_t sizesInput1[3] = {BATCH_SIZE, M, K};
  intptr_t sizesInput2[3] = {BATCH_SIZE, K, N};
  intptr_t sizesOutput[3] = {BATCH_SIZE, M, N};

  MemRef<float, 3> input1(sizesInput1, 1.0);
  MemRef<float, 3> input2(sizesInput2, 1.0);
  MemRef<float, 3> output(sizesOutput, 0.0);

  for (auto _ : state) {
    func(&input1, &input2, &output);
  }
}
} // namespace

// Register benchmark cases.
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL, Scalar, _mlir_ciface_batch_matmul_scalar)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL, AutoVectorization,
                  _mlir_ciface_batch_matmul_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL, Vectorization,
                  _mlir_ciface_batch_matmul_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL, AutoTillingVectorization,
                  _mlir_ciface_batch_matmul_tilling_vectorization)
    ->Unit(benchmark::kMillisecond);

/// Correctness Verification
void verification() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  intptr_t sizesInput1[3] = {BATCH_SIZE, M, K};
  intptr_t sizesInput2[3] = {BATCH_SIZE, K, N};
  intptr_t sizesOutput[3] = {BATCH_SIZE, M, N};

  const int input1Size = BATCH_SIZE * M * K;
  float input1Rand[input1Size];
  for (int i = 0; i < input1Size; ++i) {
    input1Rand[i] = distribution(generator);
  }
  MemRef<float, 3> input1MemRef(input1Rand, sizesInput1);

  const int input2Size = BATCH_SIZE * K * N;
  float input2Rand[input2Size];
  for (int i = 0; i < input2Size; ++i) {
    input2Rand[i] = distribution(generator);
  }
  MemRef<float, 3> input2MemRef(input2Rand, sizesInput2);

  const int outputSize = BATCH_SIZE * M * N;
  MemRef<float, 3> outputScalar(sizesOutput, 0.0);
  MemRef<float, 3> outputAutoVectorization(sizesOutput, 0.0);
  MemRef<float, 3> outputVectorization(sizesOutput, 0.0);
  MemRef<float, 3> outputTillingVectorization(sizesOutput, 0.0);

  _mlir_ciface_batch_matmul_scalar(&input1MemRef, &input2MemRef, &outputScalar);
  _mlir_ciface_batch_matmul_auto_vectorization(&input1MemRef, &input2MemRef,
                                               &outputAutoVectorization);
  _mlir_ciface_batch_matmul_vectorization(&input1MemRef, &input2MemRef,
                                          &outputVectorization);
  _mlir_ciface_batch_matmul_tilling_vectorization(&input1MemRef, &input2MemRef,
                                                  &outputTillingVectorization);

  auto resultScalar = outputScalar.getData();
  auto resultAutoVectorization = outputAutoVectorization.getData();
  auto resultVectorization = outputVectorization.getData();
  auto resultTillingVectorization = outputTillingVectorization.getData();

  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Scalar vs AutoVectorization: "
            << (areArraysEqual(resultScalar, resultAutoVectorization,
                               outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Scalar vs Vectorization: "
            << (areArraysEqual(resultScalar, resultVectorization, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Scalar vs AutoTillingVectorization: "
            << (areArraysEqual(resultScalar, resultTillingVectorization,
                               outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}

int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  verification();
  return 0;
}
