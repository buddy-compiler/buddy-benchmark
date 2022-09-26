//===- BuddyDilation2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for Dilation2D operation.
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
void _mlir_ciface_dilation_2d_constant_padding(MemRef<float, 2> *inputBuddyDilation2D,
                                           MemRef<float, 2> *kernelBuddyDilation2D,
                                           MemRef<float, 2> *outputBuddyDilation2D,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue);

void _mlir_ciface_dilation_2d_replicate_padding(MemRef<float, 2> *inputBuddyDilation2D,
                                            MemRef<float, 2> *kernelBuddyDilation2D,
                                            MemRef<float, 2> *outputBuddyDilation2D,
                                            unsigned int centerX,
                                            unsigned int centerY,
                                            unsigned int iterations,
                                            float constantValue);
}

// Declare input image and kernel.
Mat inputImageBuddyDilation2D, kernelBuddyDilation2DMat;

// Define the kernel size.
int kernelRowsBuddyDilation2D, kernelColsBuddyDilation2D;

// Define the output size.
int outputRowsBuddyDilation2D, outputColsBuddyDilation2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyDilation2D[2];
intptr_t sizesKernelBuddyDilation2D[2];
intptr_t sizesOutputBuddyDilation2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType2;

void initializeBuddyDilation2D(char **argv) {
  inputImageBuddyDilation2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelBuddyDilation2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsBuddyDilation2D = kernelBuddyDilation2DMat.rows;
  kernelColsBuddyDilation2D = kernelBuddyDilation2DMat.cols;

  outputRowsBuddyDilation2D = inputImageBuddyDilation2D.rows;
  outputColsBuddyDilation2D = inputImageBuddyDilation2D.cols;

  sizesInputBuddyDilation2D[0] = inputImageBuddyDilation2D.rows;
  sizesInputBuddyDilation2D[1] = inputImageBuddyDilation2D.cols;

  sizesKernelBuddyDilation2D[0] = kernelRowsBuddyDilation2D;
  sizesKernelBuddyDilation2D[1] = kernelColsBuddyDilation2D;

  sizesOutputBuddyDilation2D[0] = outputRowsBuddyDilation2D;
  sizesOutputBuddyDilation2D[1] = outputColsBuddyDilation2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType2 = replicate_padding;
  } else {
    BoundaryType2  = constant_padding;
  }
}

static void Buddy_Dilation2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyDilation2D(inputImageBuddyDilation2D,
                                    sizesInputBuddyDilation2D);
  MemRef<float, 2> kernelBuddyDilation2D(kernelBuddyDilation2DMat,
                                     sizesKernelBuddyDilation2D);
  MemRef<float, 2> outputBuddyDilation2D(sizesOutputBuddyDilation2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_dilation_2d_constant_padding(
          &inputBuddyDilation2D, &kernelBuddyDilation2D, &outputBuddyDilation2D,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Dilation2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyDilation2D(inputImageBuddyDilation2D,
                                    sizesInputBuddyDilation2D);
  MemRef<float, 2> kernelBuddyDilation2D(kernelBuddyDilation2DMat,
                                     sizesKernelBuddyDilation2D);
  MemRef<float, 2> outputBuddyDilation2D(sizesOutputBuddyDilation2D);

  for (auto _ : state) {
    for (int i = 0;i < state.range(0); ++i) {
      _mlir_ciface_dilation_2d_replicate_padding(
          &inputBuddyDilation2D, &kernelBuddyDilation2D, &outputBuddyDilation2D,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyDilation2D() {
  if (BoundaryType2 == replicate_padding) {
    BENCHMARK(Buddy_Dilation2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(Buddy_Dilation2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultBuddyDilation2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageBuddyDilation2D, sizesInputBuddyDilation2D);
  MemRef<float, 2> kernel(get<0>(kernelMap[argv[2]]), sizesKernelBuddyDilation2D);
  MemRef<float, 2> output(sizesOutputBuddyDilation2D);
  // Run the 2D Dilationelation.
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    _mlir_ciface_dilation_2d_replicate_padding(&input, &kernel, &output,
                                           1 /* Center X */, 1 /* Center Y */, 5,
                                           0.0f /* Constant Value */);
  } else {
    _mlir_ciface_dilation_2d_constant_padding(&input, &kernel, &output,
                                          1 /* Center X */, 1 /* Center Y */, 5,
                                          0.0f /* Constant Value */);
  }

  // Define a cv::Mat with the output of the Dilationelation.
  Mat outputImage(outputRowsBuddyDilation2D, outputColsBuddyDilation2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyDilation2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}