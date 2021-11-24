//===- Container.cpp ------------------------------------------------------===//
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
// This file implements the container descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_CONTAINER_DEF
#define UTILS_CONTAINER_DEF

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "Utils/Container.h"

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const T *data, intptr_t sizes[N], intptr_t offset) {
  static_assert(N >= 1 && N <= 4, "MemRef size not supported.");

  this->offset = offset;
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size_t size = product(sizes);
  T *ptr = new T[size];
  for (size_t i = 0; i < size; i++) {
    ptr[i] = data[i];
  }
  aligned = ptr;
  allocated = ptr;
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(intptr_t sizes[N], T init) {
  static_assert(N >= 1 && N <= 4, "MemRef size not supported.");

  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  setStrides();
  size_t size = product(sizes);
  T *data = new T[size];
  aligned = data;
  allocated = data;
  std::fill(data, data + size, init);
}

template <typename T, size_t N>
MemRef<T, N>::MemRef(cv::Mat image, intptr_t sizes[N]) {
  static_assert(N == 2 || N == 4, "Currently only support 2d and 4d memref.");

  for (size_t i = 0; i < N; i++)
    this->sizes[i] = sizes[i];
  if (N == 2) {
    // Copy image pixels for image processing memref.
    auto ptr = new T[image.rows * image.cols];
    this->allocated = ptr;
    this->aligned = ptr;
    int k = 0;
    for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
        this->aligned[k] = (T)image.at<uchar>(i, j);
        k++;
      }
    }
    setStrides();
  } else if (N == 4) {
    // Copy image pixels for deep learning tensors.
    auto ptr = new T[image.rows * image.cols * 3];
    this->allocated = ptr;
    this->aligned = ptr;
    int k = 0;
    // NHWC layout.
    for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
        for (int color = 0; color < 3; color++) {
          // Reorder to RGB layout and normalize the element.
          this->aligned[k] = (T)image.at<cv::Vec3b>(i, j)[2 - color] / 255.0f;
          k++;
        }
      }
    }
    setStrides(true);
  }
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const PNGImage &img, intptr_t sizes[N]) {
  static_assert(N == 3, "MemRef size not supported.");

  // Set the shape.
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  // Set the strides.
  setStrides();
  // Set the data.
  size_t channels = img.channels;
  size_t height = img.height;
  size_t width = img.width;
  size_t size = channels * height * width;
  T *data = new T[size];
  for (size_t h = 0; h < height; h++) {
    for (size_t w = 0; w < width * channels; w += channels) {
      for (size_t c = 0; c < channels; c++) {
        size_t offset =
            c * strides[0] + h * strides[1] + (w / channels) * strides[2];
        data[offset] = static_cast<T>(img.row_pointers[h][w + c]);
      }
    }
  }
  // Normalize image data to [0,1] range.
  if (img.color_type == PNG_COLOR_TYPE_RGB) {
    for (size_t i = 0; i < size; i++) {
      data[i] /= 255.0f;
    }
  }
  aligned = data;
  allocated = data;
}

template <typename T, std::size_t N>
MemRef<T, N>::MemRef(const std::vector<PNGImage> &imgs, intptr_t sizes[N]) {
  static_assert(N == 4, "MemRef size not supported.");

  // Set the shape.
  for (size_t i = 0; i < N; i++) {
    this->sizes[i] = sizes[i];
  }
  // Set the strides.
  setStrides();
  // Set the data.
  size_t batch = imgs.size();
  size_t channels = imgs[0].channels;
  size_t height = imgs[0].height;
  size_t width = imgs[0].width;
  size_t size = batch * channels * height * width;
  T *data = new T[size];
  for (size_t b = 0; b < batch; b++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width * channels; w += channels) {
        for (size_t c = 0; c < channels; c++) {
          size_t offset = b * strides[0] + c * strides[1] + h * strides[2] +
                          (w / channels) * strides[3];
          data[offset] = static_cast<T>(imgs[b].row_pointers[h][w + c]);
        }
      }
    }
  }
  aligned = data;
  allocated = data;
}

template <typename T, std::size_t N> MemRef<T, N>::~MemRef() {
  delete[] allocated;
}

template <typename T, std::size_t N>
void MemRef<T, N>::setStrides(const bool transpose) {
  if (transpose) {
    strides[0] = 1;
    for (long i = 1; i < N; i++) {
      strides[i] = strides[i - 1] * sizes[i - 1];
    }
  } else {
    strides[N - 1] = 1;
    for (long i = N - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }
}

template <typename T, std::size_t N>
MemRef<T, N> MemRef<T, N>::transpose(const std::vector<size_t> &axes) {
  // Check the size of the memref.
  // Currently transpose is implemented for 2d, 3d and 4d memref.
  static_assert(N >= 2 && N <= 4, "MemRef size not supported.");

  std::vector<size_t> dims = axes;
  if (dims.empty()) {
    dims.resize(N);
    std::iota(dims.rbegin(), dims.rend(), 0);
  } else {
    if (dims.size() != N) {
      throw std::runtime_error("Invalid number of axes.");
    }
    std::vector<size_t> order = axes;
    std::sort(order.begin(), order.end(),
              [](size_t i, size_t j) { return i < j; });
    if (order[0] != 0 || order[N - 1] != N - 1) {
      throw std::runtime_error("All axes must be in range [0 ... N-1].");
    }
  }

  intptr_t newSizes[N];
  for (size_t i = 0; i < N; i++) {
    newSizes[i] = sizes[dims[i]];
  }
  MemRef<T, N> res(newSizes);
  const auto newStrides = res.getStrides();
  // Copy the data.
  if (N == 2) {
    for (std::size_t h = 0; h < sizes[0]; h++) {
      for (std::size_t w = 0; w < sizes[1]; w++) {
        size_t oldOffset = h * strides[0] + w * strides[1];
        size_t newOffset = w * newStrides[0] + h * newStrides[1];
        res[newOffset] = (*this)[oldOffset];
      }
    }
  } else if (N == 3) {
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
          res[newOffset] = (*this)[oldOffset];
        }
      }
    }
  } else { // N=4.
    auto getNewDim = [&dims](size_t index, size_t n, size_t c, size_t h,
                             size_t w) {
      if (dims[index] == 0)
        return n;
      else if (dims[index] == 1)
        return c;
      else if (dims[index] == 2)
        return h;
      else
        return w;
    };
    for (std::size_t n = 0; n < sizes[0]; n++) {
      for (std::size_t c = 0; c < sizes[1]; c++) {
        for (std::size_t h = 0; h < sizes[2]; h++) {
          for (std::size_t w = 0; w < sizes[3]; w++) {
            size_t oldOffset = n * strides[0] + c * strides[1] +
                               h * strides[2] + w * strides[3];
            size_t newOffset = getNewDim(0, n, c, h, w) * newStrides[0] +
                               getNewDim(1, n, c, h, w) * newStrides[1] +
                               getNewDim(2, n, c, h, w) * newStrides[2] +
                               getNewDim(3, n, c, h, w) * newStrides[3];
            res[newOffset] = (*this)[oldOffset];
          }
        }
      }
    }
  }

  return res;
}

template <typename T, std::size_t N>
size_t MemRef<T, N>::product(intptr_t sizes[N]) const {
  size_t size = 1;
  for (size_t i = 0; i < N; i++)
    size *= sizes[i];
  return size;
}

#endif // UTILS_CONTAINER_DEF
