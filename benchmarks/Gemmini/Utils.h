//===- Utils.h ------------------------------------------------------------===//
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
// This file implements Gemmini benchmark helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef GEMMINI_UTILS_HPP
#define GEMMINI_UTILS_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------
namespace buddy {
namespace benchmark {
namespace gemmini {

int8_t *allocArrayInt8(int rows, int cols) {
  srand((unsigned int)time(NULL));
  int8_t *array = (int8_t *)malloc(rows * cols * sizeof(int8_t));
  if (array == NULL) {
    return NULL;
  }
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      array[i * cols + j] = rand() % 2;
    }
  }
  return array;
}

// Allocates a 1D array with dimensions `rows * cols` and fills it with random
// values between 0 and 99.
template <typename DATA_TYPE> DATA_TYPE *allocArray(int rows, int cols) {
  // Initialize the random number generator.
  std::srand(static_cast<unsigned int>(std::time(0)));
  // Allocate memory for the array
  DATA_TYPE *array = new DATA_TYPE[rows * cols];
  // Fill the array with random numbers between 0 and 99
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      array[i * cols + j] = static_cast<DATA_TYPE>(std::rand() % 2);
    }
  }
  return array;
}

template <typename DATA_TYPE>
void verify(DATA_TYPE *A, DATA_TYPE *B, int rows, int cols,
            const std::string &name) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";
  const double epsilon = 1e-6; // Tolerance for floating point comparison

  std::cout << name << " ";
  if (!A || !B) {
    std::cout << FAIL << " (Null pointer detected)" << std::endl;
    return;
  }

  bool isPass = true;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int k = i * cols + j;
      if (std::fabs(A[k] - B[k]) > epsilon) {
        std::cout << FAIL << std::endl;
        std::cout << "i=" << i << " j=" << j
                  << ":\tA[i][j]=" << static_cast<int>(A[k])
                  << " B[i][j]=" << static_cast<int>(B[k]) << std::endl;
        isPass = false;
        break;
      }
    }
  }
  if (isPass) {
    std::cout << PASS << std::endl;
  }
}

// By inserting assembly code to obtain clock cycles
static uint64_t readCycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

// Print a 2D array of type int8_t with dimensions `rows * cols`.
void printArrayInt8(int8_t *arr, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << static_cast<int>(arr[i * cols + j]) << " ";
    }
    std::cout << std::endl;
  }
}

} // namespace gemmini
} // namespace benchmark
} // namespace buddy

#endif // GEMMINI_UTILS_HPP
