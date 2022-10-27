//===- BuddyOpening2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for Opening2D operation.
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
void _mlir_ciface_opening_2d_constant_padding(MemRef<float, 2> *inputBuddyOpening2D,
                                           MemRef<float, 2> *kernelBuddyOpening2D,
                                           MemRef<float, 2> *outputBuddyOpening2D,
                                           MemRef<float, 2> *outputBuddyOpening2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue
    );

void _mlir_ciface_opening_2d_replicate_padding(MemRef<float, 2> *inputBuddyOpening2D,
                                           MemRef<float, 2> *kernelBuddyOpening2D,
                                           MemRef<float, 2> *outputBuddyOpening2D,
                                           MemRef<float, 2> *outputBuuddyOpening2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue

);
}

// Declare input image and kernel.
Mat inputImageBuddyOpening2D, kernelBuddyOpening2DMat;

// Define the kernel size.
int kernelRowsBuddyOpening2D, kernelColsBuddyOpening2D;

// Define the output size.
int outputRowsBuddyOpening2D, outputColsBuddyOpening2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyOpening2D[2];
intptr_t sizesKernelBuddyOpening2D[2];
intptr_t sizesOutputBuddyOpening2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType2;

void initializeBuddyOpening2D(char **argv) {
  inputImageBuddyOpening2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelBuddyOpening2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsBuddyOpening2D = kernelBuddyOpening2DMat.rows;
  kernelColsBuddyOpening2D = kernelBuddyOpening2DMat.cols;

  outputRowsBuddyOpening2D = inputImageBuddyOpening2D.rows;
  outputColsBuddyOpening2D = inputImageBuddyOpening2D.cols;

  sizesInputBuddyOpening2D[0] = inputImageBuddyOpening2D.rows;
  sizesInputBuddyOpening2D[1] = inputImageBuddyOpening2D.cols;

  sizesKernelBuddyOpening2D[0] = kernelRowsBuddyOpening2D;
  sizesKernelBuddyOpening2D[1] = kernelColsBuddyOpening2D;

  sizesOutputBuddyOpening2D[0] = outputRowsBuddyOpening2D;
  sizesOutputBuddyOpening2D[1] = outputColsBuddyOpening2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType2 = replicate_padding;
  } else {
    BoundaryType2 = constant_padding;
  }
}

static void Buddy_Opening2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyOpening2D(inputImageBuddyErosion2D,
                                    sizesInputBuddyErosion2D);
  MemRef<float, 2> kernelBuddyOpening2D(kernelBuddyErosion2DMat,
                                     sizesKernelBuddyErosion2D);
  MemRef<float, 2> outputBuddyOpening2D(sizesOutputBuddyErosion2D);
  MemRef<float, 2> outpuutBuddyOpening2D1(sizesOutputBuddyOpening2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_opening_2d_constant_padding(
          &inputBuddyOpening2D, &kernelBuddyOpening2D, &outputBuddyOpening2D, &outputBuddyOpening2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Opening2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyOpening2D(inputImageBuddyOpening2D,
                                    sizesInputBuddyOpening2D);
  MemRef<float, 2> kernelBuddyOpening2D(kernelBuddyOpening2DMat,
                                     sizesKernelBuddyOpening2D);
  MemRef<float, 2> outputBuddyOpening2D(sizesOutputBuddyOpening2D);
  MemRef<float, 2> outputBuddyOpening2D1(sizesOutputBuddyOpening2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_opening_2d_replicate_padding(
          &inputBuddyOpening2D, &kernelBuddyOpening2D, &outputBuddyOpening2D, &outputBuddyOpening2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyOpening2D() {
  if (BoundaryType2 == replicate_padding) {
    BENCHMARK(Buddy_Opening2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(Buddy_Opening2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultBuddyOpening2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageBuddyOpening2D, sizesInputBuddyOpening2D);
  MemRef<float, 2> kernel(get<0>(kernelMap[argv[2]]), sizesKernelBuddyErosion2D);
  MemRef<float, 2> output(sizesOutputBuddyOpening2D);
  MemRef<float, 2> output1(sizesOutputBuddyOpening2D);
  // Run the 2D Erosionelation.
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    _mlir_ciface_opening_2d_replicate_padding(&input, &kernel, &output, &output1,
                                           1 /* Center X */, 1 /* Center Y */, 5,
                                           0.0f /* Constant Value */);
  } else {
    _mlir_ciface_opening_2d_constant_padding(&input, &kernel, &output, &output1,
                                          1 /* Center X */, 1 /* Center Y */, 5,
                                          0.0f /* Constant Value */);
  }

  // Define a cv::Mat with the output of the Erosionelation.
  Mat outputImage(outputRowsBuddyOpening2D, outputColsBuddyOpening2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyOpening2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
