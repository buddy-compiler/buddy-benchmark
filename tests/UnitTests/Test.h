//===- Test.h -------------------------------------------------------------===//
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
// This file implements helper functions for the tests.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

// Test if:
// - two arrays of integral type (intptr_t, int, size_t, ..) are equal
// - two arrays of float or double are approximatively eqaul, to within 4 ulps (unit in the last space) from each other.
template <typename T> void ASSERT_ARRAY_EQ(const T *x, const T *y) {
  size_t size = sizeof(x) / sizeof(T);
  size_t size_y = sizeof(y) / sizeof(T);
  ASSERT_EQ(size, size_y);

  if (std::is_integral<T>::value) {
    for (size_t i = 0; i < size; i++) {
      ASSERT_EQ(x[i], y[i]);
    }
  } else if (std::is_same<T, float>::value) {
    for (size_t i = 0; i < size; i++) {
      ASSERT_FLOAT_EQ(x[i], y[i]);
    }
  } else if (std::is_same<T, double>::value) {
    for (size_t i = 0; i < size; i++) {
      ASSERT_DOUBLE_EQ(x[i], y[i]);
    }
  }
}
