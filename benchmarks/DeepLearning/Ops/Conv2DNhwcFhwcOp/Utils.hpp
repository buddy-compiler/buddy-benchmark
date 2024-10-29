//===- Utils.hpp ----------------------------------------------------------===//
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
// This file implements Conv2D NHWC-FHWC helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef CONV2D_NHWC_FHWC_UTILS_HPP
#define CONV2D_NHWC_FHWC_UTILS_HPP

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

namespace conv2d {

// Allocates a 1D array with dimensions `N * H * W * C` and fills it with random
// values between 0 and 99.
template <typename DATA_TYPE>
DATA_TYPE *allocArray(int N, int H, int W, int C) {
  // Initialize the random number generator.
  std::srand(static_cast<unsigned int>(std::time(0)));
  // Allocate memory for the array.
  int size = N * H * W * C;
  DATA_TYPE *array = new DATA_TYPE[size];
  // Fill the array with random numbers between 0 and 99.
  for (int i = 0; i < size; i++) {
    array[i] = static_cast<DATA_TYPE>(std::rand() % 100);
  }
  return array;
}

template <typename DATA_TYPE>
void verify(DATA_TYPE *A, DATA_TYPE *B, int size, const std::string &name) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";
  const double epsilon = 1e-2; // Tolerance for floating point comparison.

  std::cout << name << " ";
  if (!A || !B) {
    std::cout << FAIL << " (Null pointer detected)" << std::endl;
    return;
  }

  bool isPass = true;
  for (int i = 0; i < size; ++i) {
    if (std::fabs((A[i] - B[i]) / (B[i] == 0 ? 1 : B[i])) > epsilon) {
      std::cout << FAIL << std::endl;
      std::cout << "Index " << i << ":\tA=" << A[i] << " B=" << B[i]
                << std::endl;
      isPass = false;
      break;
    }
  }
  if (isPass) {
    std::cout << PASS << std::endl;
  }
}

} // namespace conv2d

#endif // CONV2D_NHWC_FHWC_UTILS_HPP
