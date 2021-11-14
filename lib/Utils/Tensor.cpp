//===- Tensor.cpp ---------------------------------------------------------===//
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
// This file implements the tensor descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_TENSOR_DEF
#define UTILS_TENSOR_DEF

#include "Utils/Tensor.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "Utils/PNGImage.h"

template <typename T, std::size_t N>
Tensor<T, N>::Tensor(const T *data, intptr_t offset, intptr_t sizes[N]) {
  static_assert(N >= 1 && N <= 4, "Tensor size not supported.");
  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setShape();
  setStrides();
  size_t size = batch * channels * height * width;
  allocated = new T[size];
  for (size_t i = 0; i < size; i++) {
    allocated[i] = data[i];
  }
  aligned = allocated;
}

template <typename T, std::size_t N> Tensor<T, N>::Tensor(intptr_t sizes[N]) {
  static_assert(N >= 1 && N <= 4, "Tensor size not supported.");
  offset = 0;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setShape();
  setStrides();
  size_t size = batch * channels * height * width;
  allocated = new T[size];
  aligned = allocated;
}

template <typename T, std::size_t N>
Tensor<T, N>::Tensor(const PNGImage &img, intptr_t sizes[N]) {
  static_assert(N >= 1 && N <= 4, "Image size not supported.");
  offset = 0;
  batch = 1;
  channels = img.channels;
  height = img.height;
  width = img.width;
  // Set the shape
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  // Set the strides
  setStrides();
  // Set the data
  size_t size = channels * height * width;
  allocated = new T[size];
  for (size_t h = 0; h < height; h++) {
    for (size_t w = 0; w < width * channels; w += channels) {
      for (size_t c = 0; c < channels; c++) {
        size_t offset =
            c * strides[0] + h * strides[1] + (w / channels) * strides[2];
        allocated[offset] = static_cast<T>(img.row_pointers[h][w + c]);
      }
    }
  }
  aligned = allocated;
  // Normalize image data to [0,1] range
  if (img.color_type == PNG_COLOR_TYPE_RGB) {
    for (size_t i = 0; i < size; i++) {
      allocated[i] /= 255.0f;
    }
  }
}

template <typename T, std::size_t N>
Tensor<T, N>::Tensor(const std::vector<PNGImage> &imgs, intptr_t sizes[N]) {
  static_assert(N == 4, "Images size not supported");
  offset = 0;
  batch = imgs.size();
  channels = imgs[0].channels;
  height = imgs[0].height;
  width = imgs[0].width;
  offset = 0;
  // Set the shape
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  // Set the strides
  setStrides();
  // Set the data
  size_t size = batch * channels * height * width;
  allocated = new T[size];
  for (size_t b = 0; b < batch; b++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width * channels; w += channels) {
        for (size_t c = 0; c < channels; c++) {
          size_t offset = b * strides[0] + c * strides[1] + h * strides[2] +
                          (w / channels) * strides[3];
          allocated[offset] = static_cast<T>(imgs[b].row_pointers[h][w + c]);
        }
      }
    }
  }
  aligned = allocated;
}

template <typename T, std::size_t N> Tensor<T, N>::~Tensor() {
  if (allocated == aligned) {
    if (allocated)
      delete[] allocated;
  } else {
    if (allocated)
      delete[] allocated;
    if (aligned)
      delete[] aligned;
  }
}

template <typename T, std::size_t N> void Tensor<T, N>::setStrides() {
  strides[N - 1] = 1;
  for (long i = N - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
}

template <typename T, std::size_t N> void Tensor<T, N>::setShape() {
  if (N == 4) { // NCHW
    batch = sizes[0];
    channels = sizes[1];
    height = sizes[2];
    width = sizes[3];
  } else if (N == 3) { // CHW
    batch = 1;
    channels = sizes[0];
    height = sizes[1];
    width = sizes[2];
  } else if (N == 2) { // HW
    batch = 1;
    channels = 1;
    height = sizes[0];
    width = sizes[1];
  } else { // W
    batch = 1;
    channels = 1;
    height = 1;
    width = sizes[0];
  }
}

template <typename T, std::size_t N>
Tensor<T, N> Tensor<T, N>::transpose(const std::vector<size_t> &dims) {
  if (dims.size() != N) {
    throw std::runtime_error("Invalid dims size.");
  }
  auto axes = dims;
  std::sort(axes.begin(), axes.end(), [](size_t i, size_t j) { return i < j; });
  if (axes[0] != 0 || axes[N - 1] != N - 1) {
    throw std::runtime_error("All dimensions must be in range 0..N-1");
  }

  intptr_t newSizes[N];
  for (size_t i = 0; i < N; i++) {
    newSizes[i] = sizes[dims[i]];
  }
  Tensor<T, N> t(newSizes);
  auto newStrides = t.getStrides();
  static_assert(N == 3, "Currently only support 3D tensor.");
  // Copy the data
  if (N == 3) {
    auto getNewDim = [&dims](size_t index, size_t c, size_t h, size_t w) {
      if (dims[index] == 0)
        return c;
      else if (dims[index] == 1)
        return h;
      else
        return w;
    };
    for (std::size_t c = 0; c < sizes[0]; c++) {
      for (std::size_t h = 0; h < sizes[1]; h++) {
        for (std::size_t w = 0; w < sizes[2]; w++) {
          size_t oldOffset = c * strides[0] + h * strides[1] + w * strides[2];
          size_t newOffset = getNewDim(0, c, h, w) * newStrides[0] +
                             getNewDim(1, c, h, w) * newStrides[1] +
                             getNewDim(2, c, h, w) * newStrides[2];
          t.at(newOffset) = this->at(oldOffset);
        }
      }
    }
  }

  return t;
}

#endif // UTILS_TENSOR_DEF
