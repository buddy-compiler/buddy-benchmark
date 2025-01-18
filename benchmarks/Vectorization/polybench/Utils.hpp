//===- Utils.cpp ----------------------------------------------------------===//
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

#ifndef POLYBENCH_UTILS_HPP
#define POLYBENCH_UTILS_HPP

#include <iostream>
#include <string>

namespace polybench {

// Mimic the behavior of POLYBENCH_DUMP_START macro.
void startDump();

// Mimic the behavior of POLYBENCH_DUMP_FINISH macro.
void finishDump();

// Mimic the behavior of POLYBENCH_DUMP_BEGIN macro.
void beginDump(const std::string &name);

// Mimic the behavior of POLYBENCH_DUMP_END macro.
void endDump(const std::string &name);

// Get the name of a dataset size by its ID.
std::string getPolybenchDatasetSizeName(int size_id);

// Get the ID of a dataset size by its name.
int getPolybenchDatasetSizeID(const std::string &name);

// Verification function. Derived from DeepLearning benchmark.
template <typename DATA_TYPE>
void verify(DATA_TYPE *A, DATA_TYPE *B, int size, const std::string &name) {
  const std::string PASS = "\033[32mPASS\033[0m";
  const std::string FAIL = "\033[31mFAIL\033[0m";

  std::cout << name << " ";
  if (!A || !B) {
    std::cout << FAIL << " (Null pointer detected)" << std::endl;
    return;
  }

  bool isPass = true;
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

} // namespace polybench

#endif // POLYBENCH_UTILS_HPP