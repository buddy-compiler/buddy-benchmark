//===------------ BuddyRotate2DBenchmark.cpp ------------------------------===//
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
// This file implements the benchmark for Rotate2D operation.
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

// Declare the input name.
std::string inputNameBuddyRotate2D;

// Define sizes of input and output.
intptr_t sizesInputBuddyRotate2D[2];
intptr_t sizesOutputBuddyRotate2D[2];

void initializeBuddyRotate2D(char **argv,
                             Img<float, 2> inputImageBuddyRotate2D) {
  inputNameBuddyRotate2D = argv[1];

  sizesInputBuddyRotate2D[0] = inputImageBuddyRotate2D.getSizes()[0];
  sizesInputBuddyRotate2D[1] = inputImageBuddyRotate2D.getSizes()[1];

  float sinAngle = std::sin(angleRad);
  float cosAngle = std::cos(angleRad);

  sizesOutputBuddyRotate2D[0] =
      std::round(std::abs(sizesInputBuddyRotate2D[0] * cosAngle) +
                 std::abs(sizesInputBuddyRotate2D[1] * sinAngle));
  sizesOutputBuddyRotate2D[1] =
      std::round(std::abs(sizesInputBuddyRotate2D[0] * sinAngle) +
                 std::abs(sizesInputBuddyRotate2D[1] * cosAngle));
}

// Benchmarking functions.
static void Buddy_Rotate2D(benchmark::State &state) {
  // Define the MemRef descriptor for input and output.
  Img<float, 2> inputBuddyRotate2D =
      dip::imread<float, 2>(inputNameBuddyRotate2D, dip::IMGRD_GRAYSCALE);

  MemRef<float, 2> outputBuddyRotate2D(sizesOutputBuddyRotate2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Rotate2D function.
      dip::detail::_mlir_ciface_rotate_2d(&inputBuddyRotate2D, angleRad,
                                          &outputBuddyRotate2D);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyRotate2D() {
  BENCHMARK(Buddy_Rotate2D)->Arg(1)->Unit(benchmark::kMillisecond);
}

// Generate result image.
void generateResultBuddyRotate2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptors for output.
  MemRef<float, 2> output(sizesOutputBuddyRotate2D);

  // Run the 2D rotate function.
  dip::detail::_mlir_ciface_rotate_2d(&input, angleRad, &output);

  intptr_t sizes[2] = {output.getSizes()[0], output.getSizes()[1]};

  // Define Img containers for output images.
  Img<float, 2> outputImage(output.getData(), sizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyRotate2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}