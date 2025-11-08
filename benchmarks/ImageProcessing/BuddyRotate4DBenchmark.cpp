//===------------ BuddyRotate4DBenchmark.cpp ------------------------------===//
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
// This file implements the benchmark for Rotate4D operation.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/imgcodecs/loadsave.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const float angleRad = M_PI * 30.0 / 180;
const int inputBatch = 3;

// Define the input image.
Img<float, 4> inputBuddyRotate4D_NHWC;
Img<float, 4> inputBuddyRotate4D_NCHW;

// Declare the input name.
std::string inputNameBuddyRotate4D;

// Define sizes of input and output.
intptr_t sizesInputBuddyRotate4D_NHWC[4];
intptr_t sizesOutputBuddyRotate4D_NHWC[4];
intptr_t sizesInputBuddyRotate4D_NCHW[4];
intptr_t sizesOutputBuddyRotate4D_NCHW[4];

void initializeBuddyRotate4D(char **argv) {
  inputNameBuddyRotate4D = argv[1];

  // Read as color image in [HWC] format.
  Img<float, 3> inputImage = dip::imread<float, 3>(argv[1], dip::IMGRD_COLOR);
  const int inputHeight = inputImage.getSizes()[0];
  const int inputWidth = inputImage.getSizes()[1];
  const int inputChannel = inputImage.getSizes()[2];

  sizesInputBuddyRotate4D_NHWC[0] = sizesInputBuddyRotate4D_NCHW[0] =
      inputBatch;
  sizesInputBuddyRotate4D_NHWC[1] = sizesInputBuddyRotate4D_NCHW[2] =
      inputHeight;
  sizesInputBuddyRotate4D_NHWC[2] = sizesInputBuddyRotate4D_NCHW[3] =
      inputWidth;
  sizesInputBuddyRotate4D_NHWC[3] = sizesInputBuddyRotate4D_NCHW[1] =
      inputChannel;

  const int inputStride = inputImage.getSize();

  Img<float, 4> inputImages_NHWC(sizesInputBuddyRotate4D_NHWC);
  auto imagePtr = inputImages_NHWC.getData();
  memcpy(imagePtr, inputImage.getData(), inputStride * sizeof(float));
  for (int i = 1; i < inputBatch; i++) {
    Img<float, 3> input = dip::imread<float, 3>(argv[1], dip::IMGRD_COLOR);
    memcpy(imagePtr + i * inputStride, input.getData(),
           inputStride * sizeof(float));
  }

  float sinAngle = std::sin(angleRad);
  float cosAngle = std::cos(angleRad);
  const int outputHeight =
      std::round(std::abs(inputImage.getSizes()[0] * cosAngle) +
                 std::abs(inputImage.getSizes()[1] * sinAngle));
  const int outputWidth =
      std::round(std::abs(inputImage.getSizes()[0] * sinAngle) +
                 std::abs(inputImage.getSizes()[1] * cosAngle));

  sizesOutputBuddyRotate4D_NHWC[0] = sizesOutputBuddyRotate4D_NCHW[0] =
      inputBatch;
  sizesOutputBuddyRotate4D_NHWC[1] = sizesOutputBuddyRotate4D_NCHW[2] =
      outputHeight;
  sizesOutputBuddyRotate4D_NHWC[2] = sizesOutputBuddyRotate4D_NCHW[3] =
      outputWidth;
  sizesOutputBuddyRotate4D_NHWC[3] = sizesOutputBuddyRotate4D_NCHW[1] =
      inputChannel;

  inputBuddyRotate4D_NHWC = inputImages_NHWC;

  Img<float, 4> inputImages_NCHW(sizesInputBuddyRotate4D_NCHW);
  dip::detail::Transpose<float, 4>(&inputImages_NCHW, &inputImages_NHWC,
                                   {0, 3, 1, 2});
  inputBuddyRotate4D_NCHW = inputImages_NCHW;
}

// Benchmarking functions.
static void Buddy_Rotate4D_NHWC(benchmark::State &state) {
  MemRef<float, 4> outputBuddyRotate4D(sizesOutputBuddyRotate4D_NHWC);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Rotate4D function.
      dip::detail::_mlir_ciface_rotate_4d_nhwc(&inputBuddyRotate4D_NHWC,
                                               angleRad, &outputBuddyRotate4D);
    }
  }
}

static void Buddy_Rotate4D_NCHW(benchmark::State &state) {
  MemRef<float, 4> outputBuddyRotate4D(sizesOutputBuddyRotate4D_NCHW);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Rotate4D function.
      dip::detail::_mlir_ciface_rotate_4d_nchw(&inputBuddyRotate4D_NCHW,
                                               angleRad, &outputBuddyRotate4D);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyRotate4D() {
  BENCHMARK(Buddy_Rotate4D_NHWC)->Arg(1)->Unit(benchmark::kMillisecond);
  BENCHMARK(Buddy_Rotate4D_NCHW)->Arg(1)->Unit(benchmark::kMillisecond);
}

// Generate result image.
void generateResultBuddyRotate4D(char **argv) {
  // Define the MemRef descriptors for output.
  MemRef<float, 4> output_NHWC(sizesOutputBuddyRotate4D_NHWC);

  // Run the 4D rotate function.
  dip::detail::_mlir_ciface_rotate_4d_nhwc(&inputBuddyRotate4D_NHWC, angleRad,
                                           &output_NHWC);

  intptr_t imageSizes[3] = {output_NHWC.getSizes()[1],
                            output_NHWC.getSizes()[2],
                            output_NHWC.getSizes()[3]};
  Img<float, 3> image_NHWC(output_NHWC.getData(), imageSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyRotate4D_NHWC.png", image_NHWC);

  MemRef<float, 4> output_NCHW(sizesOutputBuddyRotate4D_NCHW);

  dip::detail::_mlir_ciface_rotate_4d_nchw(&inputBuddyRotate4D_NCHW, angleRad,
                                           &output_NCHW);

  dip::detail::Transpose<float, 4>(&output_NHWC, &output_NCHW, {0, 2, 3, 1});

  Img<float, 3> image_NCHW(output_NHWC.getData(), imageSizes);

  result = dip::imwrite("ResultBuddyRotate4D_NCHW.png", image_NCHW);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}