//===- BuddyErosion2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for Erosion2D operation.
//
//===----------------------------------------------------------------------===//

#include "ImageProcessing/Kernels.h"
#include "Utils/Container.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_erosion_2d_constant_padding(MemRef<float, 2> *inputBuddyErosion2D,
                                           MemRef<float, 2> *kernelBuddyErosion2D,
                                           MemRef<float, 2> *outputBuddyErosion2D,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue);

void _mlir_ciface_erosion_2d_replicate_padding(MemRef<float, 2> *inputBuddyErosion2D,
                                            MemRef<float, 2> *kernelBuddyErosion2D,
                                            MemRef<float, 2> *outputBuddyErosion2D,
                                            unsigned int centerX,
                                            unsigned int centerY,
                                            unsigned int iterations,
                                            float constantValue);
}

// Declare input image and kernel.
Mat inputImageBuddyErosion2D, kernelBuddyErosion2DMat;

// Define the kernel size.
int kernelRowsBuddyErosion2D, kernelColsBuddyErosion2D;

// Define the output size.
int outputRowsBuddyErosion2D, outputColsBuddyErosion2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyErosion2D[2];
intptr_t sizesKernelBuddyErosion2D[2];
intptr_t sizesOutputBuddyErosion2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType1;

void initializeBuddyErosion2D(char **argv) {
  inputImageBuddyErosion2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelBuddyErosion2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsBuddyErosion2D = kernelBuddyErosion2DMat.rows;
  kernelColsBuddyErosion2D = kernelBuddyErosion2DMat.cols;

  outputRowsBuddyErosion2D = inputImageBuddyErosion2D.rows;
  outputColsBuddyErosion2D = inputImageBuddyErosion2D.cols;

  sizesInputBuddyErosion2D[0] = inputImageBuddyErosion2D.rows;
  sizesInputBuddyErosion2D[1] = inputImageBuddyErosion2D.cols;

  sizesKernelBuddyErosion2D[0] = kernelRowsBuddyErosion2D;
  sizesKernelBuddyErosion2D[1] = kernelColsBuddyErosion2D;

  sizesOutputBuddyErosion2D[0] = outputRowsBuddyErosion2D;
  sizesOutputBuddyErosion2D[1] = outputColsBuddyErosion2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType1 = replicate_padding;
  } else {
    BoundaryType1 = constant_padding;
  }
}

static void Buddy_Erosion2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyErosion2D(inputImageBuddyErosion2D,
                                    sizesInputBuddyErosion2D);
  MemRef<float, 2> kernelBuddyErosion2D(kernelBuddyErosion2DMat,
                                     sizesKernelBuddyErosion2D);
  MemRef<float, 2> outputBuddyErosion2D(sizesOutputBuddyErosion2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_erosion_2d_constant_padding(
          &inputBuddyErosion2D, &kernelBuddyErosion2D, &outputBuddyErosion2D,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Erosion2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyErosion2D(inputImageBuddyErosion2D,
                                    sizesInputBuddyErosion2D);
  MemRef<float, 2> kernelBuddyErosion2D(kernelBuddyErosion2DMat,
                                     sizesKernelBuddyErosion2D);
  MemRef<float, 2> outputBuddyErosion2D(sizesOutputBuddyErosion2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_erosion_2d_replicate_padding(
          &inputBuddyErosion2D, &kernelBuddyErosion2D, &outputBuddyErosion2D,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyErosion2D() {
  if (BoundaryType1 == replicate_padding) {
    BENCHMARK(Buddy_Erosion2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(Buddy_Erosion2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultBuddyErosion2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageBuddyErosion2D, sizesInputBuddyErosion2D);
  MemRef<float, 2> kernel(get<0>(kernelMap[argv[2]]), sizesKernelBuddyErosion2D);
  MemRef<float, 2> output(sizesOutputBuddyErosion2D);
  // Run the 2D Erosionelation.
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    _mlir_ciface_erosion_2d_replicate_padding(&input, &kernel, &output,
                                           1 /* Center X */, 1 /* Center Y */, 5,
                                           0.0f /* Constant Value */);
  } else {
    _mlir_ciface_erosion_2d_constant_padding(&input, &kernel, &output,
                                          1 /* Center X */, 1 /* Center Y */, 5,
                                          0.0f /* Constant Value */);
  }

  // Define a cv::Mat with the output of the Erosionelation.
  Mat outputImage(outputRowsBuddyErosion2D, outputColsBuddyErosion2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyErosion2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
