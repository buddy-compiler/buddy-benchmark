//===- Correctness.h --------------------------------------------------------===//
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
// Correctness checking for containers.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_CORRECTNESS
#define UTILS_CORRECTNESS

#include <memory>
#include <vector>
#include <cmath>
#include <functional>
#include <utility>

#include "Utils/Container.h"
#include "DTW.h"

template<typename T>
T euclideanDistMono(std::function<T(int)> acc1, std::function<T(int)> acc2, int count) {
  T sum = 0;
  for (int i = 0; i < count; ++i) {
    T dist = acc1(i) - acc2(i);
    sum += dist * dist;
  }
  return sqrt(sum);
}

// First: sum of all errors, Second: maximum error found
template<typename T>
std::pair<T,T> errorCheckMono(std::function<T(int)> acc1, std::function<T(int)> acc2, int count, T beta = 0.01f) {
  T sum = 0;
  T max = 0;
  for (int i = 0; i < count; ++i) {
    T dist = abs(acc1(i) - acc2(i));
    sum += (dist > beta)?dist:0;
    if (dist>max) max=dist;
  }
  return {sum,max};
}

// P: p-norm (2 for euclidean, 1 for manhattan)
template<typename T>
double DTWCheckMono(std::function<T(int)> acc1, std::function<T(int)> acc2, int count, int p = 2) {
  std::vector<std::vector<double>> vec1;
  std::vector<std::vector<double>> vec2;
  for (int i = 0; i < count; ++i) {
    vec1.push_back(std::vector<double>{acc1(i)});
    vec2.push_back(std::vector<double>{acc2(i)});
  }
  return DTW::dtw_distance_only(vec1, vec2, p);
}


#endif // UTILS_CORRECTNESS
