//===- boost_gil_conv2d.cpp -------------------------------------------===//
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
// This file implements an example for depicting the use of Boost GIL's 2D Convolution API.
//
//===----------------------------------------------------------------------===//

#include "ImageProcessing/Kernels.h"
#include <boost/gil.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

namespace gil = boost::gil;

void fill_gil_view_with_opencv_mat(cv::Mat opencv_mat,
                                   gil::gray8_view_t gil_view) {
  for (int i = 0; i < opencv_mat.rows; i++)
    for (int j = 0; j < opencv_mat.cols; j++)
      gil_view(j, i) = opencv_mat.at<uchar>(i, j);
}

int main(int argc, char *argv[]) {
  // Read input image using opencv's imread()
  cv::Mat opencv_image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  // Declare input image
  gil::gray8_image_t image(opencv_image.cols, opencv_image.rows);

  // Fill GIL image view with image read using opencv's imread()
  fill_gil_view_with_opencv_mat(opencv_image, gil::view(image));

  // Declare output image
  gil::gray8_image_t output(image.dimensions());

  // Create a 2D GIL kernel
  gil::detail::kernel_2d<float> kernel(sobel3x3KernelAlign, 9, 1, 1);

  clock_t start, end;
  start = clock();
  // Apply 2D convolution between input image and kernel
  gil::detail::convolve_2d(gil::view(image), kernel, gil::view(output));
  end = clock();
  std::cout << "Execution time: " << (double)(end - start) / CLOCKS_PER_SEC
            << " s" << std::endl;

  // Save obtained image
  gil::write_view(argv[2], gil::view(output), gil::png_tag{});
}
