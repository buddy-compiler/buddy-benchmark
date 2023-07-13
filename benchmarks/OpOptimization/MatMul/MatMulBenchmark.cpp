//===- MatMulBenchmark.cpp ------------------------------------------------===//
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
// This file implements the benchmark for MatMul operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

// Define target layout.
#define N 64
#define M 3136
#define K 576

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
// Declare the matmul C interface.
extern "C" {
void _mlir_ciface_matmul_ocv(MemRef<float, 2> *A, MemRef<float, 2> *B,
                             MemRef<float, 2> *C);
void _mlir_ciface_matmul_scalar(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                MemRef<float, 2> *C);
}

void BM_MATMUL_OCV(benchmark::State &state) {
  intptr_t sizesA[2] = {M, K};
  intptr_t sizesB[2] = {K, N};
  intptr_t sizesC[2] = {M, N};

  MemRef<float, 2> A(sizesA, 1.0);
  MemRef<float, 2> B(sizesB, 1.0);
  MemRef<float, 2> C(sizesC, 0);

  for (auto _ : state) {
    _mlir_ciface_matmul_ocv(&A, &B, &C);
  }
}

void BM_MATMUL_SCALAR(benchmark::State &state) {
  intptr_t sizesA[2] = {M, K};
  intptr_t sizesB[2] = {K, N};
  intptr_t sizesC[2] = {M, N};

  MemRef<float, 2> A(sizesA, 1.0);
  MemRef<float, 2> B(sizesB, 1.0);
  MemRef<float, 2> C(sizesC, 0);

  for (auto _ : state) {
    _mlir_ciface_matmul_scalar(&A, &B, &C);
  }
}
} // namespace

// Register benchmark cases.
BENCHMARK(BM_MATMUL_OCV)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_SCALAR)->Unit(benchmark::kMillisecond);

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
  std::uniform_int_distribution<int> distribution(1, 100);

  // Set the layout sizes of input and output memref container.
  intptr_t sizesA[2] = {M, K};
  intptr_t sizesB[2] = {K, N};
  intptr_t sizesC[2] = {M, N};

  // Generate input A and input B memref container with random numbers.
  const int inputASize = M * K;
  float inputARand[inputASize];
  for (int i = 0; i < inputASize; ++i) {
    inputARand[i] = distribution(generator);
  }
  MemRef<float, 2> inputAMemRef(inputARand, sizesA);

  const int inputBSize = K * N;
  float inputBRand[inputBSize];
  for (int i = 0; i < inputBSize; ++i) {
    inputBRand[i] = distribution(generator);
  }
  MemRef<float, 2> inputBMemRef(inputBRand, sizesB);

  // Generate output memref container with zero.
  const int outputSize = M * N;
  MemRef<float, 2> outputScalar(sizesC, 0);
  MemRef<float, 2> outputOCV(sizesC, 0);

  // Perform all the matmul implementation.
  _mlir_ciface_matmul_scalar(&inputAMemRef, &inputBMemRef, &outputScalar);
  _mlir_ciface_matmul_ocv(&inputAMemRef, &inputBMemRef, &outputOCV);

  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultOCV = outputOCV.getData();

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "OCV case: "
            << (areArraysEqual(resultScalar, resultOCV, outputSize) ? PASS
                                                                    : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}
