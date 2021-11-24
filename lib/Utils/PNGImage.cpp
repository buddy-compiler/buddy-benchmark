//===- PNGImage.cpp ------------------------------------------------------===//
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
// This file implements the PNG image io.
//
//===----------------------------------------------------------------------===//

#include "Utils/PNGImage.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

bool PNGImage::readpng(const std::string &filePath) {
  FILE *file = fopen(filePath.c_str(), "rb");
  if (!file) {
    return false;
  }
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr) {
    fclose(file);
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    fclose(file);
    return false;
  }
  png_infop end_info = png_create_info_struct(png_ptr);
  if (!end_info) {
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    fclose(file);
    return false;
  }
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    fclose(file);
    return false;
  }

  png_init_io(png_ptr, file);
  png_read_info(png_ptr, info_ptr);

  // read width.
  width = png_get_image_width(png_ptr, info_ptr);
  // read height.
  height = png_get_image_height(png_ptr, info_ptr);
  // read channels.
  channels = png_get_channels(png_ptr, info_ptr);

  // [TODO] Support other color types (RGBA, GRAY, ...).
  color_type = png_get_color_type(png_ptr, info_ptr);
  if (color_type != PNG_COLOR_TYPE_RGB) {
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    fclose(file);
    return false;
  }

  // Allocate an array of pointers.
  row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
  // Allocate memory for each element of row_pointers.
  for (size_t h = 0; h < height; h++) {
    row_pointers[h] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
  }
  png_read_image(png_ptr, row_pointers);

  // Clean.
  fclose(file);
  png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

  return true;
}

PNGImage::PNGImage(const std::string &filepath) {
  if (!readpng(filepath)) {
    throw std::runtime_error("Error reading png file.");
  }
}

PNGImage::~PNGImage() {
  for (size_t h = 0; h < height; h++) {
    free(row_pointers[h]);
  }
  free(row_pointers);
}
