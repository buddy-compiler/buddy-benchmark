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

#include <stdint.h>
#include <vector>

#ifdef AUDIO_CONTAINER
#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#include <kfr/io.hpp>
#endif

#ifdef IMAGE_CONTAINER
#include <opencv2/opencv.hpp>
#include "Utils/PNGImage.h"
#endif

// ToDo : Identify the proper usecase which requires "TRANSPOSE" operation.

enum class IMAGE_MATRIX_OPERATION {
  DEFAULT,
  NORMALIZE,
  NORMALIZE_AND_TRANSPOSE,
  TRANSPOSE
};
#endif
// MemRef descriptor.
// - T represents the type of the elements.
// - N represents the number of dimensions.
// - The storage order is NCHW.
template <typename T, size_t N> class MemRef {
public:
  // Constructor from data.
  MemRef(const T *data, intptr_t sizes[N], intptr_t offset = 0);
  // Constructor from shape.
  MemRef(intptr_t sizes[N], T init = T(0));
  MemRef(std::vector<size_t> sizes, T init = T(0));
  #ifdef IMAGE_CONTAINER
  // Create a memref from an opencv image.
  MemRef(cv::Mat image, intptr_t sizes[N],
         IMAGE_MATRIX_OPERATION operation = IMAGE_MATRIX_OPERATION::DEFAULT);
  // Constructor from a png image.
  MemRef(const PNGImage &img, intptr_t sizes[N],
         IMAGE_MATRIX_OPERATION operation = IMAGE_MATRIX_OPERATION::DEFAULT);
  // Constructor from a vector of png images.
  // Assume that all the images have the same shape.
  MemRef(const std::vector<PNGImage> &imgs, intptr_t sizes[N],
         IMAGE_MATRIX_OPERATION operation = IMAGE_MATRIX_OPERATION::DEFAULT);
  #endif
  #ifdef AUDIO_CONTAINER
  // Constructor from KFR univector. 
  MemRef(const kfr::univector<T>, intptr_t sizes[N]);
  // Constructor from KFR univector2d.
  // Data layout is determined by sizes[N]. 
  MemRef(const kfr::univector2d<T>, intptr_t sizes[N]);
  #endif
  // Copy constructor.
  MemRef(const MemRef<T, N> &other);
  // Copy assignment operator.
  MemRef<T, N> &operator=(const MemRef<T, N> &other);
  // Move constructor.
  MemRef(MemRef<T, N> &&other) noexcept;
  // Move assignment operator.
  MemRef<T, N> &operator=(MemRef<T, N> &&other) noexcept;
  // Desctrutor.
  ~MemRef();
  // Permute the dimensions.
  // Reorder the dimensions from {0, 1, ..., N-1} to {N-1, ..., 1, 0} when axes
  // is empty.
  MemRef<T, N> transpose(const std::vector<size_t> &axes = {});
  // Get the data pointer.
  T *getData() { return aligned; }
  // Get the sizes (shape).
  const intptr_t *getSizes() { return sizes; }
  // Get the strides.
  const intptr_t *getStrides() { return strides; }
  // Get the rank of the memref.
  size_t getRank() const { return N; }
  // Get the size (number of elements).
  size_t getSize() const { return size; }
  // Get the element at index.
  const T &operator[](size_t index) const { return aligned[index + offset]; }
  T &operator[](size_t index) { return aligned[index + offset]; }

private:
  // Set the strides.
  // Computes the strides of the transposed tensor for transpose=true.
  void setStrides(const bool transpose = false);
  // Compute the product of array elements.
  size_t product(intptr_t sizes[N]) const;

  // Data.
  T *allocated;
  T *aligned;
  // Offset.
  intptr_t offset = 0;
  // Shape.
  intptr_t sizes[N];
  // Strides.
  intptr_t strides[N];
  // Number of elements.
  size_t size;
};

#include "Utils/Container.cpp"

#endif // UTILS_CONTAINER
