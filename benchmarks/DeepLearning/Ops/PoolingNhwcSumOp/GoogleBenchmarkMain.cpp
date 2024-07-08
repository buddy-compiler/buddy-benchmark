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
// This file implements the benchmark for sum pooling (nhwc) operation.
//
//===----------------------------------------------------------------------===//

#include <buddy/Core/Container.h>
#include <iostream>
#include <random>
#include <sys/time.h>

// Define target layout.
#define INPUT_N 1
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_C 16
#define FILTER_H 4
#define FILTER_W 4
#define OUTPUT_N 1
#define OUTPUT_C 16
#define OUTPUT_H 29
#define OUTPUT_W 29

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
// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_pooling_nhwc_sum_scalar(MemRef<float, 4> *input,
                                          MemRef<float, 2> *filter,
                                          MemRef<float, 4> *output);
void _mlir_ciface_pooling_nhwc_sum_auto_vectorization(MemRef<float, 4> *input,
                                                      MemRef<float, 2> *filter,
                                                      MemRef<float, 4> *output);
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
  std::uniform_int_distribution<int> distribution(0.0, 1.0);

  // Set the layout sizes of input and output memref container.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_H, INPUT_W, INPUT_C};
  intptr_t sizesFilter[2] = {FILTER_H, FILTER_W};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_C, OUTPUT_H, OUTPUT_W};

  // Generate input A and input B memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_H * INPUT_W * INPUT_C;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }
  MemRef<float, 4> inputMemRef(inputRand, sizesInput);

  const int filterSize = FILTER_H * FILTER_W;
  float filterRand[filterSize];
  for (int i = 0; i < filterSize; ++i) {
    filterRand[i] = distribution(generator);
  }
  MemRef<float, 2> filterMemRef(filterRand, sizesFilter);

  // Generate output memref container with zero.
  const int outputSize = OUTPUT_N * OUTPUT_C * OUTPUT_H * OUTPUT_W;
  MemRef<float, 4> outputScalar(sizesOutput, 0.0);
  MemRef<float, 4> outputAutoVectorization(sizesOutput, 0.0);

  double StartTime, EndTime;
  StartTime = rtclock();
  _mlir_ciface_pooling_nhwc_sum_scalar(&inputMemRef, &filterMemRef,
                                       &outputScalar);
  EndTime = rtclock();
  // Output the result
  std::cout << "Total time running pooling_nhwc_sum scalar: "
            << EndTime - StartTime << " s." << std::endl;

  StartTime = rtclock();
  _mlir_ciface_pooling_nhwc_sum_auto_vectorization(&inputMemRef, &filterMemRef,
                                                   &outputAutoVectorization);
  EndTime = rtclock();
  // Output the result
  std::cout << "Total time running pooling_nhwc_sum auto vectorization: "
            << EndTime - StartTime << " s." << std::endl;

  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultAutoVectorization = outputAutoVectorization.getData();

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultAutoVectorization,
                               outputSize)
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
