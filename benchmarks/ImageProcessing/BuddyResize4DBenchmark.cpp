//===------------ BuddyResize4DBenchmark.cpp ------------------------------===//
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
// This file implements the benchmark for Resize4D operation.
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
std::string inputNameBuddyResize4D;

// Define sizes of input and output.
intptr_t sizesInputBuddyResize4D[4];
intptr_t sizesOutputBuddyResize4D[4] = {1, 100, 250, 3};

void initializeBuddyResize4D(char **argv,
                             Img<float, 4> inputImageBuddyResize4D) {
  inputNameBuddyResize4D = argv[1];
  sizesInputBuddyResize4D[0] = inputImageBuddyResize4D.getSizes()[0];
  sizesInputBuddyResize4D[1] = inputImageBuddyResize4D.getSizes()[1];
  sizesInputBuddyResize4D[2] = inputImageBuddyResize4D.getSizes()[2];
  sizesInputBuddyResize4D[3] = inputImageBuddyResize4D.getSizes()[3];
}

// Benchmarking functions.
static void
Buddy_Resize4D_Nearest_Neighbour_Interpolation(benchmark::State &state) {
  // Define the MemRef descriptor for input and output.
  Img<float, 3> inputImg =
      dip::imread<float, 3>(inputNameBuddyResize4D, dip::IMGRD_COLOR);
  intptr_t sizes[4] = {1, inputImg.getSizes()[0], inputImg.getSizes()[1],
                       inputImg.getSizes()[2]};
  Img<float, 4> inputBuddyResize4D(inputImg.getData(), sizes);
  MemRef<float, 4> outputBuddyResize4D(sizesOutputBuddyResize4D);

  std::vector<float> scalingRatios(2);
  scalingRatios[1] =
      inputBuddyResize4D.getSizes()[0] * 1.0f / sizesOutputBuddyResize4D[1];
  scalingRatios[0] =
      inputBuddyResize4D.getSizes()[1] * 1.0f / sizesOutputBuddyResize4D[2];

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Resize2D function.
      dip::detail::_mlir_ciface_resize_4d_nearest_neighbour_interpolation(
          &inputBuddyResize4D, scalingRatios[0], scalingRatios[1],
          &outputBuddyResize4D);
    }
  }
}

static void Buddy_Resize4D_Bilinear_Interpolation(benchmark::State &state) {
  // Define the MemRef descriptor for input and output.
  Img<float, 3> inputImg =
      dip::imread<float, 3>(inputNameBuddyResize4D, dip::IMGRD_COLOR);
  intptr_t sizes[4] = {1, inputImg.getSizes()[0], inputImg.getSizes()[1],
                       inputImg.getSizes()[2]};
  Img<float, 4> inputBuddyResize4D(inputImg.getData(), sizes);
  MemRef<float, 4> outputBuddyResize4D(sizesOutputBuddyResize4D);

  std::vector<float> scalingRatios(2);
  scalingRatios[1] =
      inputBuddyResize4D.getSizes()[0] * 1.0f / sizesOutputBuddyResize4D[1];
  scalingRatios[0] =
      inputBuddyResize4D.getSizes()[1] * 1.0f / sizesOutputBuddyResize4D[2];

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Resize2D function.
      dip::detail::_mlir_ciface_resize_4d_bilinear_interpolation(
          &inputBuddyResize4D, scalingRatios[0], scalingRatios[1],
          &outputBuddyResize4D);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyResize4D() {
  BENCHMARK(Buddy_Resize4D_Nearest_Neighbour_Interpolation)
      ->Arg(1)
      ->Unit(benchmark::kMillisecond);
  BENCHMARK(Buddy_Resize4D_Bilinear_Interpolation)
      ->Arg(1)
      ->Unit(benchmark::kMillisecond);
}

// Generate result image.
void generateResultBuddyResize4D(char **argv, Img<float, 4> input) {
  // Define the MemRef descriptors for output.
  MemRef<float, 4> output_nearest_neighbour_interpolation(
      sizesOutputBuddyResize4D);
  MemRef<float, 4> output_bilinear_interpolation(sizesOutputBuddyResize4D);

  std::vector<float> scalingRatios(2);
  scalingRatios[1] = input.getSizes()[1] * 1.0f / sizesOutputBuddyResize4D[1];
  scalingRatios[0] = input.getSizes()[2] * 1.0f / sizesOutputBuddyResize4D[2];

  // Run the 2D resize function.
  dip::detail::_mlir_ciface_resize_4d_nearest_neighbour_interpolation(
      &input, scalingRatios[0], scalingRatios[1],
      &output_nearest_neighbour_interpolation);

  dip::detail::_mlir_ciface_resize_4d_bilinear_interpolation(
      &input, scalingRatios[0], scalingRatios[1],
      &output_bilinear_interpolation);

  // Define Img containers for output images.
  Img<float, 4> outputNearestNeighbourInterpolation(
      output_nearest_neighbour_interpolation.getData(),
      sizesOutputBuddyResize4D);
  Img<float, 4> outputBilinearInterpolation(
      output_bilinear_interpolation.getData(), sizesOutputBuddyResize4D);

  // Define Img with the output of Resize4D.
  intptr_t outputImageNearestNeighbourInterpolationSizes[3] = {
      outputNearestNeighbourInterpolation.getSizes()[1],
      outputNearestNeighbourInterpolation.getSizes()[2],
      outputNearestNeighbourInterpolation.getSizes()[3]};
  Img<float, 3> outputImageNearestNeighbourInterpolation(
      outputNearestNeighbourInterpolation.getData(),
      outputImageNearestNeighbourInterpolationSizes);

  intptr_t outputImageBilinearInterpolationSizes[3] = {
      outputBilinearInterpolation.getSizes()[1],
      outputBilinearInterpolation.getSizes()[2],
      outputBilinearInterpolation.getSizes()[3]};
  Img<float, 3> outputImageBilinearInterpolation(
      outputBilinearInterpolation.getData(),
      outputImageBilinearInterpolationSizes);

  // Write output to PNG.
  bool result =
      dip::imwrite("ResultBuddyResize4D_NearestNeighbourInterpolation.png",
                   outputImageNearestNeighbourInterpolation);
  result |= dip::imwrite("ResultBuddyResize4D_BilinearInterpolation.png",
                         outputImageBilinearInterpolation);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
