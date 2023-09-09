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

#include <array>
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

// Define target layout.
#define M 64
#define N 3136
#define K 576
#define BATCH_M 128
#define BATCH_N 784
#define BATCH_K 72
#define BATCH 16

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
void _mlir_ciface_matmul_transform(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
void _mlir_ciface_matmul_broadcast_16(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                      MemRef<float, 2> *C);
void _mlir_ciface_matmul_broadcast_32(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                      MemRef<float, 2> *C);
void _mlir_ciface_matmul_broadcast_64(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                      MemRef<float, 2> *C);
void _mlir_ciface_matmul_broadcast_128(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                       MemRef<float, 2> *C);
void _mlir_ciface_matmul_broadcast_256(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                       MemRef<float, 2> *C);
void _mlir_ciface_matmul_scalar(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                MemRef<float, 2> *C);
void _mlir_ciface_batch_matmul_scalar(MemRef<float, 3> *A, MemRef<float, 3> *B,
                                      MemRef<float, 3> *C);
void _mlir_ciface_batch_matmul_broadcast_64(MemRef<float, 3> *A,
                                            MemRef<float, 3> *B,
                                            MemRef<float, 3> *C);
void _mlir_ciface_batch_matmul_broadcast_64_omp(MemRef<float, 3> *A,
                                                MemRef<float, 3> *B,
                                                MemRef<float, 3> *C);
}

#define DEFINE_MATMUL_BENCHMARK(name, func)                                    \
  void BM_MATMUL_##name(benchmark::State &state) {                             \
    intptr_t sizesA[2] = {M, K};                                               \
    intptr_t sizesB[2] = {K, N};                                               \
    intptr_t sizesC[2] = {M, N};                                               \
                                                                               \
    MemRef<float, 2> A(sizesA, 1.0);                                           \
    MemRef<float, 2> B(sizesB, 1.0);                                           \
    MemRef<float, 2> C(sizesC, 0);                                             \
                                                                               \
    for (auto _ : state) {                                                     \
      func(&A, &B, &C);                                                        \
    }                                                                          \
  }

#define DEFINE_BATCH_MATMUL_BENCHMARK(name, func)                              \
  void BM_BATCH_MATMUL_##name(benchmark::State &state) {                       \
    intptr_t sizesA[3] = {BATCH, BATCH_M, BATCH_K};                            \
    intptr_t sizesB[3] = {BATCH, BATCH_K, BATCH_N};                            \
    intptr_t sizesC[3] = {BATCH, BATCH_M, BATCH_N};                            \
                                                                               \
    MemRef<float, 3> A(sizesA, 1.0);                                           \
    MemRef<float, 3> B(sizesB, 1.0);                                           \
    MemRef<float, 3> C(sizesC, 0);                                             \
                                                                               \
    for (auto _ : state) {                                                     \
      func(&A, &B, &C);                                                        \
    }                                                                          \
  }

DEFINE_MATMUL_BENCHMARK(OCV, _mlir_ciface_matmul_ocv)
DEFINE_MATMUL_BENCHMARK(TRANSFORM, _mlir_ciface_matmul_transform)
DEFINE_MATMUL_BENCHMARK(BROADCAST_16, _mlir_ciface_matmul_broadcast_16)
DEFINE_MATMUL_BENCHMARK(BROADCAST_32, _mlir_ciface_matmul_broadcast_32)
DEFINE_MATMUL_BENCHMARK(BROADCAST_64, _mlir_ciface_matmul_broadcast_64)
DEFINE_MATMUL_BENCHMARK(BROADCAST_128, _mlir_ciface_matmul_broadcast_128)
DEFINE_MATMUL_BENCHMARK(BROADCAST_256, _mlir_ciface_matmul_broadcast_256)
DEFINE_MATMUL_BENCHMARK(SCALAR, _mlir_ciface_matmul_scalar)
DEFINE_BATCH_MATMUL_BENCHMARK(SCALAR, _mlir_ciface_batch_matmul_scalar)
DEFINE_BATCH_MATMUL_BENCHMARK(BROADCAST_64,
                              _mlir_ciface_batch_matmul_broadcast_64)
DEFINE_BATCH_MATMUL_BENCHMARK(BROADCAST_64_OMP,
                              _mlir_ciface_batch_matmul_broadcast_64_omp)
} // namespace

// Register benchmark cases.
BENCHMARK(BM_MATMUL_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_OCV)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_TRANSFORM)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_BROADCAST_16)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_BROADCAST_32)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_BROADCAST_64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_BROADCAST_128)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_BROADCAST_256)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MATMUL_BROADCAST_256)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BATCH_MATMUL_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BATCH_MATMUL_BROADCAST_64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BATCH_MATMUL_BROADCAST_64_OMP)->Unit(benchmark::kMillisecond);

// Correctness Verification
// The verification does not affect the performance.
// - Set the scalar case as the criteria.
// - Input elements are random numbers.
// - Output elements are initialized to zero.
// - Compare the output of various optimizations with the scalar version to
//   verify correctness.
void matmul_verification() {
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
  MemRef<float, 2> outputTransform(sizesC, 0);
  MemRef<float, 2> outputBroadcast16(sizesC, 0);
  MemRef<float, 2> outputBroadcast32(sizesC, 0);
  MemRef<float, 2> outputBroadcast64(sizesC, 0);
  MemRef<float, 2> outputBroadcast128(sizesC, 0);
  MemRef<float, 2> outputBroadcast256(sizesC, 0);

  // Perform all the matmul implementation.
  _mlir_ciface_matmul_scalar(&inputAMemRef, &inputBMemRef, &outputScalar);
  _mlir_ciface_matmul_ocv(&inputAMemRef, &inputBMemRef, &outputOCV);
  _mlir_ciface_matmul_transform(&inputAMemRef, &inputBMemRef, &outputTransform);
  _mlir_ciface_matmul_broadcast_16(&inputAMemRef, &inputBMemRef,
                                   &outputBroadcast16);
  _mlir_ciface_matmul_broadcast_32(&inputAMemRef, &inputBMemRef,
                                   &outputBroadcast32);
  _mlir_ciface_matmul_broadcast_64(&inputAMemRef, &inputBMemRef,
                                   &outputBroadcast64);
  _mlir_ciface_matmul_broadcast_128(&inputAMemRef, &inputBMemRef,
                                    &outputBroadcast128);
  _mlir_ciface_matmul_broadcast_256(&inputAMemRef, &inputBMemRef,
                                    &outputBroadcast256);

  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultOCV = outputOCV.getData();
  auto resultTransform = outputTransform.getData();
  auto resultBroadcast16 = outputBroadcast16.getData();
  auto resultBroadcast32 = outputBroadcast32.getData();
  auto resultBroadcast64 = outputBroadcast64.getData();
  auto resultBroadcast128 = outputBroadcast128.getData();
  auto resultBroadcast256 = outputBroadcast256.getData();

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "OCV case: "
            << (areArraysEqual(resultScalar, resultOCV, outputSize) ? PASS
                                                                    : FAIL)
            << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultTransform, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Broadcast 16 case: "
            << (areArraysEqual(resultScalar, resultBroadcast16, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Broadcast 32 case: "
            << (areArraysEqual(resultScalar, resultBroadcast32, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Broadcast 64 case: "
            << (areArraysEqual(resultScalar, resultBroadcast64, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Broadcast 128 case: "
            << (areArraysEqual(resultScalar, resultBroadcast128, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Broadcast 256 case: "
            << (areArraysEqual(resultScalar, resultBroadcast256, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}

void batch_matmul_verification() {
  // Set the random number generator.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> distribution(1, 100);

  // Set the layout sizes of input and output memref container.
  intptr_t sizesA[3] = {BATCH, BATCH_M, BATCH_K};
  intptr_t sizesB[3] = {BATCH, BATCH_K, BATCH_N};
  intptr_t sizesC[3] = {BATCH, BATCH_M, BATCH_N};

  // Generate input A and input B memref container with random numbers.
  const int inputASize = BATCH * (BATCH_M) * (BATCH_K);
  // float inputARand[inputASize];
  auto inputARand = new std::array<float, inputASize>();
  for (int i = 0; i < inputASize; ++i) {
    (*inputARand)[i] = distribution(generator);
  }
  MemRef<float, 3> inputAMemRef(inputARand->data(), sizesA);

  const int inputBSize = BATCH * (BATCH_K) * (BATCH_N);
  // float inputBRand[inputBSize];
  auto inputBRand = new std::array<float, inputBSize>();
  for (int i = 0; i < inputBSize; ++i) {
    (*inputBRand)[i] = distribution(generator);
  }
  MemRef<float, 3> inputBMemRef(inputBRand->data(), sizesB);

  // Generate output memref container with zero.
  const int outputSize = BATCH * (BATCH_M) * (BATCH_N);
  MemRef<float, 3> outputScalar(sizesC, 0);
  MemRef<float, 3> outputBroadcast64(sizesC, 0);
  MemRef<float, 3> outputBroadcast64OMP(sizesC, 0);

  // Perform all the matmul implementation.
  _mlir_ciface_batch_matmul_scalar(&inputAMemRef, &inputBMemRef, &outputScalar);
  _mlir_ciface_batch_matmul_broadcast_64(&inputAMemRef, &inputBMemRef,
                                         &outputBroadcast64);
  _mlir_ciface_batch_matmul_broadcast_64_omp(&inputAMemRef, &inputBMemRef,
                                             &outputBroadcast64OMP);

  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultBroadcast64 = outputBroadcast64.getData();
  auto resultBroadcast64OMP = outputBroadcast64OMP.getData();

  // Print the verfication result.
  std::cout << "Batch Matmul Broadcast 64 case: "
            << (areArraysEqual(resultScalar, resultBroadcast64,
                               outputSize / BATCH)
                    ? PASS
                    : FAIL)
            << std::endl;

  std::cout << "Batch Matmul Broadcast 64 OpenMP case: "
            << (areArraysEqual(resultScalar, resultBroadcast64OMP,
                               outputSize / BATCH)
                    ? PASS
                    : FAIL)
            << std::endl;

  std::cout << "-----------------------------------------------------------"
            << std::endl;
}
