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
// This is the main file of the matmul benchmark.
//
//===----------------------------------------------------------------------===//

#include <buddy/Core/Container.h>
#include <iostream>
#include <random>
#include <sys/time.h>

// Define target layout.
#define M 64
#define N 3136
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

double rtclock() {
  struct timeval tp;
  int stat = gettimeofday(&tp, nullptr);
  if (stat != 0)
    fprintf(stderr, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
}
} // namespace

namespace {
// Declare the matmul C interface.
extern "C" {
void _mlir_ciface_matmul_scalar(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                MemRef<float, 2> *C);
void _mlir_ciface_matmul_transform(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
}
} // namespace

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
  MemRef<float, 2> outputTransform(sizesC, 0);

  // Perform all the matmul implementation.
  double StartTime, EndTime;
  StartTime = rtclock();
  _mlir_ciface_matmul_scalar(&inputAMemRef, &inputBMemRef, &outputScalar);
  EndTime = rtclock();
  // Output the result
  std::cout << "Total time running matmul scalar: " << EndTime - StartTime
            << " s." << std::endl;

  StartTime = rtclock();
  _mlir_ciface_matmul_transform(&inputAMemRef, &inputBMemRef, &outputTransform);
  EndTime = rtclock();
  // Output the result
  std::cout << "Total time running matmul transform: " << EndTime - StartTime
            << " s." << std::endl;

  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultTransform = outputTransform.getData();

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultTransform, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}

int main(int argc, char **argv) {
  // Run correctness verification.
  verification();
  return 0;
}
