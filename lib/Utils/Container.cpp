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

#include <memory>
#include "Utils/Container.h"

template <int Dim>
MemRef<Dim>::MemRef(intptr_t rows, intptr_t cols, float *aligned,
                    intptr_t offset, intptr_t sizes[Dim],
                    intptr_t strides[Dim]) {
  auto ptr = new float[rows * cols];
  this->allocated = ptr;
  this->aligned = ptr;
  int k = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      this->aligned[k] = aligned[k];
      k++;
    }
  }
  this->offset = offset;
  for (int i = 0; i < Dim; i++)
    this->sizes[i] = sizes[i];
  for (int j = 0; j < Dim; j++)
    this->strides[j] = strides[j];
}

template <int Dim>
MemRef<Dim>::MemRef(cv::Mat image, intptr_t offset, intptr_t sizes[Dim],
                    intptr_t strides[Dim]) {
  // Copy image pixels for image processing memref.
  if (Dim == 2) {
    auto ptr = new float[image.rows * image.cols];
    this->allocated = ptr;
    this->aligned = ptr;
    int k = 0;
    for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
        this->aligned[k] = (float)image.at<uchar>(i, j);
        k++;
      }
    }
  }
  // Copy image pixels for deep learning tensors.
  if (Dim == 4) {
    auto ptr = new float[image.rows * image.cols * 3];
    this->allocated = ptr;
    this->aligned = ptr;
    int k = 0;
    // NHWC layout.
    for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
        for (int color = 0; color < 3; color++) {
          // Reorder to RGB layout.
          this->aligned[k] = (float)image.at<cv::Vec3b>(i, j)[2 - color];
          k++;
        }
      }
    }
  }
  this->offset = offset;
  for (int i = 0; i < Dim; i++)
    this->sizes[i] = sizes[i];
  for (int j = 0; j < Dim; j++)
    this->strides[j] = strides[j];
}

template <int Dim>
MemRef<Dim>::MemRef(intptr_t rows, intptr_t cols, intptr_t offset,
                    intptr_t sizes[Dim], intptr_t strides[Dim]) {
  auto ptr = new float[rows * cols];
  this->allocated = ptr;
  this->aligned = ptr;
  this->offset = offset;
  for (int i = 0; i < Dim; i++)
    this->sizes[i] = sizes[i];
  for (int j = 0; j < Dim; j++)
    this->strides[j] = strides[j];
}

// Constructor for deep learning output.
template <int Dim>
MemRef<Dim>::MemRef(intptr_t results, intptr_t offset, intptr_t sizes[Dim],
                    intptr_t strides[Dim]) {
  auto ptr = new float[results];
  this->allocated = ptr;
  this->aligned = ptr;
  for (int i = 0; i < results; i++) {
    this->aligned[i] = 0;
  }
  this->offset = offset;
  for (int i = 0; i < Dim; i++)
    this->sizes[i] = sizes[i];
  for (int j = 0; j < Dim; j++)
    this->strides[j] = strides[j];
}

template <int Dim> MemRef<Dim>::~MemRef() { delete[] this->allocated; }

#endif // UTILS_CONTAINER_DEF
