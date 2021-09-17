//===- BuddyContainer.h ---------------------------------------------------===//
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
// Memref descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef IMAGE_PROCESSING_BUDDY_CONTAINER
#define IMAGE_PROCESSING_BUDDY_CONTAINER

#include <opencv2/opencv.hpp>

// Define Memref Descriptor.
struct MemRef {
public:
  MemRef(intptr_t rows, intptr_t cols, float *aligned, intptr_t offset,
         intptr_t sizes[2], intptr_t strides[2]);
  MemRef(cv::Mat image, intptr_t offset, intptr_t sizes[2],
         intptr_t strides[2]);
  MemRef(intptr_t rows, intptr_t cols, intptr_t offset, intptr_t sizes[2],
         intptr_t strides[2]);
  ~MemRef();
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
};

#endif // MAGE_PROCESSING_BUDDY_CONTAINER
