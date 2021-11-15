//===- Container.h --------------------------------------------------------===//
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
// Container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_CONTAINER
#define UTILS_CONTAINER

#include <memory>
#include <opencv2/opencv.hpp>
#include <stdint.h>

template <typename T, size_t Dim> struct MemRef {
public:
  MemRef(intptr_t rows, intptr_t cols, T *aligned, intptr_t offset,
         intptr_t sizes[Dim], intptr_t strides[Dim]);
  MemRef(T init, intptr_t sizes[Dim], intptr_t strides[Dim]);
  MemRef(cv::Mat image, intptr_t offset, intptr_t sizes[Dim],
         intptr_t strides[Dim]);
  MemRef(intptr_t rows, intptr_t cols, intptr_t offset, intptr_t sizes[Dim],
         intptr_t strides[Dim]);
  MemRef(intptr_t results, intptr_t offset, intptr_t sizes[Dim],
         intptr_t strides[Dim]);
  ~MemRef();
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[Dim];
  intptr_t strides[Dim];
};

#include "Utils/Container.cpp"

#endif // UTILS_CONTAINER
