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
#include <vector>

#include "Utils/PNGImage.h"

// MemRef descriptor
// - T represents the type of the elements
// - N represents the number of dimensions
// - The storage order is NCHW
template <typename T, size_t N> class MemRef {
public:
   // Constructor from data
  MemRef(const T *data, intptr_t sizes[N], intptr_t offset = 0);
   // Constructor from shape
  MemRef(intptr_t sizes[N], T init = T(0));
   // Create a memref from an opencv image
  MemRef(cv::Mat image, intptr_t sizes[N]);
  // Constructor from a png image
   MemRef(const PNGImage &img, intptr_t sizes[N]);
  // Constructor from a vector of png images
  // Assume that all the images have the same shape
  MemRef(const std::vector<PNGImage> &imgs, intptr_t sizes[N]);
  // Desctrutor
  ~MemRef();
  // Permute the dimensions
  // Reorder the dimensions from {0, 1, ..., N-1} to {N-1, ..., 1, 0} when axes
  // is empty
  MemRef<T, N> transpose(const std::vector<size_t> &axes = {});
  // Get the data pointer
  T *getData() { return allocated; }
  // Get the sizes
  const intptr_t *getSizes() { return sizes; }
  // Get the strides
  const intptr_t *getStrides() { return strides; }
  // Get the element at index
  const T &operator[](size_t index) const { return allocated[index + offset]; }
  T &operator[](size_t index) { return allocated[index + offset]; }

private:
   // Set the strides
   // Computes the strides of the transposed tensor for transpose=true
   void setStrides(const bool transpose = false);
   // Compute the product of array elements
   size_t product(intptr_t sizes[N]) const;

   // Data
  T *allocated;
  T *aligned;
   // Offset
  intptr_t offset = 0;
   // Shape
  intptr_t sizes[N];
   // Strides
  intptr_t strides[N];
};

#include "Utils/Container.cpp"

#endif // UTILS_CONTAINER
