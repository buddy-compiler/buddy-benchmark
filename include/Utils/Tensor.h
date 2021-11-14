//===- Tensor.h -----------------------------------------------------------===//
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
// Tensor / Ranked Memref descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_TENSOR
#define UTILS_TENSOR

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Utils/PNGImage.h"

// Generic tensor/ranked memref data structure
// The storage order is NCHW
template <typename T, std::size_t N> class Tensor {
public:
  // Constructors
  Tensor(const T *data, intptr_t offset, intptr_t sizes[N]);
  Tensor(intptr_t sizes[N]);
  // Constructor from an image
  Tensor(const PNGImage &img, intptr_t sizes[N]);
  // Constructor from images
  // Assume that all the images have the same shape
  Tensor(const std::vector<PNGImage> &imgs, intptr_t sizes[N]);
  // Desctrutor
  ~Tensor();
  // Permute the dimensions of the tensor
  Tensor<T, N> transpose(const std::vector<size_t> &);
  // Get the data pointer
  T *getData() { return allocated; }
  // Get the sizes
  const intptr_t *getSizes() { return sizes; }
  // Get the strides
  const intptr_t *getStrides() { return strides; }
  // Get the element at index
  T at(size_t index) const { return allocated[index]; }
  T &at(size_t index) { return allocated[index]; }

  // Set the strides from the shape
  void setStrides();
  // Set the shape
  void setShape();

private:
  // Data
  T *allocated;
  T *aligned;
  // Offset
  intptr_t offset;
  // Shape of the tensor
  intptr_t sizes[N];
  // Strides
  intptr_t strides[N];
  // Shape of the image
  intptr_t batch;
  intptr_t channels;
  intptr_t height;
  intptr_t width;
};

#include "Utils/Tensor.cpp"

#endif // UTILS_TENSOR
