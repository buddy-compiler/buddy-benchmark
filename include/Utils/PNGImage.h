//===- PNGImage.h ---------------------------------------------------------===//
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
// PNG Image IO.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_PNGIMAGE
#define UTILS_PNGIMAGE

#include <cstddef>
#include <string>

#include "png.h"

class PNGImage {
private:
  // Color type of the image
  png_byte color_type;
  // Shape of the image
  size_t channels;
  size_t height;
  size_t width;
  // Data
  png_bytep *row_pointers = nullptr;

  // Read a png image
  bool readpng(const std::string &filePath);

public:
  // Constructor
  PNGImage() = delete;
  PNGImage(const std::string &filePath);
  // Destructor
  ~PNGImage(){};

  template <typename T, std::size_t> friend class Tensor;
};

#include "Utils/PNGImage.cpp"

#endif // UTILS_PNGIMAGE
