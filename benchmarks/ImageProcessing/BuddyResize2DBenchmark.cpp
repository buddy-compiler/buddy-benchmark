//===------------ BuddyResize2DBenchmark.cpp ------------------------------===//
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
// This file implements the benchmark for Resize2D operation.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/imgcodecs/loadsave.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the input name.
std::string inputNameBuddyResize2D;

// Define sizes of input and output.
intptr_t sizesInputBuddyResize2D[2];
intptr_t sizesOutputBuddyResize2D[2] = {100, 250};

void initializeBuddyResize2D(char **argv,
                             Img<float, 2> inputImageBuddyResize2D) {
  inputNameBuddyResize2D = argv[1];

  sizesInputBuddyResize2D[0] = inputImageBuddyResize2D.getSizes()[0];
  sizesInputBuddyResize2D[1] = inputImageBuddyResize2D.getSizes()[1];
}

// Benchmarking functions.
static void
Buddy_Resize2D_Nearest_Neighbour_Interpolation(benchmark::State &state) {
  // Define the MemRef descriptor for input and output.
  Img<float, 2> inputBuddyResize2D =
      dip::imread<float, 2>(inputNameBuddyResize2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyResize2D(sizesOutputBuddyResize2D);

  std::vector<float> scalingRatios(2);
  scalingRatios[1] =
      inputBuddyResize2D.getSizes()[0] * 1.0f / sizesOutputBuddyResize2D[0];
  scalingRatios[0] =
      inputBuddyResize2D.getSizes()[1] * 1.0f / sizesOutputBuddyResize2D[1];

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Resize2D function.
      dip::detail::_mlir_ciface_resize_2d_nearest_neighbour_interpolation(
          &inputBuddyResize2D, scalingRatios[0], scalingRatios[1],
          &outputBuddyResize2D);
    }
  }
}

static void Buddy_Resize2D_Bilinear_Interpolation(benchmark::State &state) {
  // Define the MemRef descriptor for input and output.
  Img<float, 2> inputBuddyResize2D =
      dip::imread<float, 2>(inputNameBuddyResize2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> outputBuddyResize2D(sizesOutputBuddyResize2D);

  std::vector<float> scalingRatios(2);
  scalingRatios[1] =
      inputBuddyResize2D.getSizes()[0] * 1.0f / sizesOutputBuddyResize2D[0];
  scalingRatios[0] =
      inputBuddyResize2D.getSizes()[1] * 1.0f / sizesOutputBuddyResize2D[1];

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Resize2D function.
      dip::detail::_mlir_ciface_resize_2d_bilinear_interpolation(
          &inputBuddyResize2D, scalingRatios[0], scalingRatios[1],
          &outputBuddyResize2D);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyResize2D() {
  BENCHMARK(Buddy_Resize2D_Nearest_Neighbour_Interpolation)
      ->Arg(1)
      ->Unit(benchmark::kMillisecond);
  BENCHMARK(Buddy_Resize2D_Bilinear_Interpolation)
      ->Arg(1)
      ->Unit(benchmark::kMillisecond);
}

// Generate result image.
void generateResultBuddyResize2D(char **argv, Img<float, 2> input) {
  // Define the MemRef descriptors for output.
  MemRef<float, 2> output_nearest_neighbour_interpolation(
      sizesOutputBuddyResize2D);
  MemRef<float, 2> output_bilinear_interpolation(sizesOutputBuddyResize2D);

  std::vector<float> scalingRatios(2);
  scalingRatios[1] = input.getSizes()[0] * 1.0f / sizesOutputBuddyResize2D[0];
  scalingRatios[0] = input.getSizes()[1] * 1.0f / sizesOutputBuddyResize2D[1];

  // Run the 2D resize function.
  dip::detail::_mlir_ciface_resize_2d_nearest_neighbour_interpolation(
      &input, scalingRatios[0], scalingRatios[1],
      &output_nearest_neighbour_interpolation);

  dip::detail::_mlir_ciface_resize_2d_bilinear_interpolation(
      &input, scalingRatios[0], scalingRatios[1],
      &output_bilinear_interpolation);

  // Define Img containers for output images.
  Img<float, 2> outputImageNearestNeighbourInterpolation(
      output_nearest_neighbour_interpolation.getData(),
      sizesOutputBuddyResize2D);
  Img<float, 2> outputImageBilinearInterpolation(
      output_bilinear_interpolation.getData(), sizesOutputBuddyResize2D);

  // Write output to PNG.
  bool result =
      dip::imwrite("ResultBuddyResize2D_NearestNeighbourInterpolation.png",
                   outputImageNearestNeighbourInterpolation);
  result |= dip::imwrite("ResultBuddyResize2D_BilinearInterpolation.png",
                         outputImageBilinearInterpolation);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
