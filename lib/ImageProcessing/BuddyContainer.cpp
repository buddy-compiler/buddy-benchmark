//===- BuddyContainer.cpp -------------------------------------------------===//
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
// This file implements the MemRef descriptor.
//
//===----------------------------------------------------------------------===//

#include "ImageProcessing/BuddyContainer.h"

MemRef::MemRef(intptr_t rows, intptr_t cols, float *aligned, intptr_t offset,
               intptr_t sizes[2], intptr_t strides[2]) {
  this->allocated = new float[1];
  this->aligned = new float[rows * cols];
  int k = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      this->aligned[k] = aligned[k];
      k++;
    }
  }
  this->offset = offset;
  for (int i = 0; i < 2; i++)
    this->sizes[i] = sizes[i];
  for (int j = 0; j < 2; j++)
    this->strides[j] = strides[j];
}

MemRef::MemRef(cv::Mat image, intptr_t offset, intptr_t sizes[2],
               intptr_t strides[2]) {
  this->allocated = new float[1];
  this->aligned = new float[image.rows * image.cols];
  int k = 0;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      this->aligned[k] = (float)image.at<uchar>(i, j);
      k++;
    }
  }
  this->offset = offset;
  for (int i = 0; i < 2; i++)
    this->sizes[i] = sizes[i];
  for (int j = 0; j < 2; j++)
    this->strides[j] = strides[j];
}

MemRef::MemRef(intptr_t rows, intptr_t cols, intptr_t offset, intptr_t sizes[2],
               intptr_t strides[2]) {
  this->allocated = new float[1];
  this->aligned = new float[rows * cols];
  this->offset = offset;
  for (int i = 0; i < 2; i++)
    this->sizes[i] = sizes[i];
  for (int j = 0; j < 2; j++)
    this->strides[j] = strides[j];
}

MemRef::~MemRef() {
  delete[] this->allocated;
  delete[] this->aligned;
}
