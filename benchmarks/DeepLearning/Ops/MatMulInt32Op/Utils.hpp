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
// This file implements MatMul helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef MATMUL_UTILS_HPP
#define MATMUL_UTILS_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

namespace matmul {

// Allocates a 1D array with dimensions `rows * cols` and fills it with random
// integer values between -500 and 500.
template <typename DATA_TYPE> DATA_TYPE *allocArray(int rows, int cols) {
  // Initialize the random number generator.
  std::srand(static_cast<unsigned int>(std::time(0)));
  // Allocate memory for the array.
  int size = rows * cols;
  DATA_TYPE *array = new DATA_TYPE[size];
  // Fill the array with random numbers between -500 and 500.
  for (int i = 0; i < size; ++i) {
    array[i] = std::rand() % 1001 - 500;
  }
  return array;
}

template <typename DATA_TYPE>
void verify(DATA_TYPE *A, DATA_TYPE *B, int rows, int cols,
            const std::string &name) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";

  std::cout << name << " ";
  if (!A || !B) {
    std::cout << FAIL << " (Null pointer detected)" << std::endl;
    return;
  }

  bool isPass = true;
  int size = rows * cols;
  for (int i = 0; i < size; ++i) {
    if (A[i] != B[i]) {
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

} // namespace matmul

#endif // MATMUL_UTILS_HPP
