//===- BuddyClosing2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for Closing2D operation.
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
void _mlir_ciface_closing_2d_constant_padding(MemRef<float, 2> *inputBuddyClosing2D,
                                           MemRef<float, 2> *kernelBuddyClosing2D,
                                           MemRef<float, 2> *outputBuddyClosing2D,
                                           MemRef<float, 2> *outputBuddyClosing2D1,
                                           MemRef<float, 2> * copyMemRefClosing2D,
                                           MemRef<float, 2>* copyMemRefClosing2D1,
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue
    );

void _mlir_ciface_closing_2d_replicate_padding(MemRef<float, 2> *inputBuddyClosing2D,
                                           MemRef<float, 2> *kernelBuddyClosing2D,
                                           MemRef<float, 2> *outputBuddyClosing2D,
                                           MemRef<float, 2> *outputBuuddyClosing2D1,
                                           MemRef<float, 2> * copyMemRefClosing2D,
                                           MemRef<float, 2>* copyMemRefClosing2D1,                                           
                                           unsigned int centerX,
                                           unsigned int centerY,
                                           unsigned int iterations,
                                           float constantValue

);
}

// Declare input image and kernel.
Mat inputImageBuddyClosing2D, kernelBuddyClosing2DMat;

// Define the kernel size.
int kernelRowsBuddyClosing2D, kernelColsBuddyClosing2D;

// Define the output size.
int outputRowsBuddyClosing2D, outputColsBuddyClosing2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyClosing2D[2];
intptr_t sizesKernelBuddyClosing2D[2];
intptr_t sizesOutputBuddyClosing2D[2];

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption BoundaryType4;

void initializeClosing2D(char **argv) {
  inputImageBuddyClosing2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelBuddyClosing2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsBuddyClosing2D = kernelBuddyClosing2DMat.rows;
  kernelColsBuddyClosing2D = kernelBuddyClosing2DMat.cols;

  outputRowsBuddyClosing2D = inputImageBuddyClosing2D.rows;
  outputColsBuddyClosing2D = inputImageBuddyClosing2D.cols;

  sizesInputBuddyClosing2D[0] = inputImageBuddyClosing2D.rows;
  sizesInputBuddyClosing2D[1] = inputImageBuddyClosing2D.cols;

  sizesKernelBuddyClosing2D[0] = kernelRowsBuddyClosing2D;
  sizesKernelBuddyClosing2D[1] = kernelColsBuddyClosing2D;

  sizesOutputBuddyClosing2D[0] = outputRowsBuddyClosing2D;
  sizesOutputBuddyClosing2D[1] = outputColsBuddyClosing2D;

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    BoundaryType4 = replicate_padding;
  } else {
    BoundaryType4 = constant_padding;
  }
}

static void Buddy_Closing2D_Constant_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyClosing2D(inputImageBuddyClosing2D,
                                    sizesInputBuddyClosing2D);
  MemRef<float, 2> kernelBuddyClosing2D(kernelBuddyClosing2DMat,
                                     sizesKernelBuddyClosing2D);
  MemRef<float, 2> outputBuddyClosing2D(sizesOutputBuddyClosing2D);
  MemRef<float, 2> outputBuddyClosing2D1(sizesOutputBuddyClosing2D);
  MemRef<float, 2> copyMemRefClosing2D(sizesOutputBuddyClosing2D, -1.f);
  MemRef<float, 2> copyMemRefClosing2D1(sizesOutputBuddyClosing2D, 256.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_closing_2d_constant_padding(
          &inputBuddyClosing2D, &kernelBuddyClosing2D, &outputBuddyClosing2D, &outputBuddyClosing2D1, &copyMemRefClosing2D, &copyMemRefClosing2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

static void Buddy_Closing2D_Replicate_Padding(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyClosing2D(inputImageBuddyClosing2D,
                                    sizesInputBuddyClosing2D);
  MemRef<float, 2> kernelBuddyClosing2D(kernelBuddyClosing2DMat,
                                     sizesKernelBuddyClosing2D);
  MemRef<float, 2> outputBuddyClosing2D(sizesOutputBuddyClosing2D);
  MemRef<float, 2> outputBuddyClosing2D1(sizesOutputBuddyClosing2D);
  MemRef<float, 2> copyMemRefClosing2D(sizesOutputBuddyClosing2D, -1.f);
  MemRef<float, 2> copyMemRefClosing2D1(sizesOutputBuddyClosing2D, 256.f);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_closing_2d_replicate_padding(
          &inputBuddyClosing2D, &kernelBuddyClosing2D, &outputBuddyClosing2D, &outputBuddyClosing2D1, &copyMemRefClosing2D, &copyMemRefClosing2D1,
          1 /* Center X */, 1 /* Center Y */,5, 0.0f /* Constant Value */);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyClosing2D() {
  if (BoundaryType4 == replicate_padding) {
    BENCHMARK(Buddy_Closing2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(Buddy_Closing2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultBuddyClosing2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageBuddyClosing2D, sizesInputBuddyClosing2D);
  MemRef<float, 2> kernel(get<0>(kernelMap[argv[2]]), sizesKernelBuddyClosing2D);
  MemRef<float, 2> output(sizesOutputBuddyClosing2D);
  MemRef<float, 2> output1(sizesOutputBuddyClosing2D);
  MemRef<float, 2> copymemref(sizesOutputBuddyClosing2D, -1.f);
  MemRef<float, 2> copymemref1(sizesOutputBuddyClosing2D, 256.f);
  // Run the 2D Closing operation
  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    _mlir_ciface_closing_2d_replicate_padding(&input, &kernel, &output, &output1, &copymemref, &copymemref1,
                                           1 /* Center X */, 1 /* Center Y */, 5,
                                           0.0f /* Constant Value */);
  } else {
    _mlir_ciface_closing_2d_constant_padding(&input, &kernel, &output, &output1, &copymemref, &copymemref1,
                                          1 /* Center X */, 1 /* Center Y */, 5,
                                          0.0f /* Constant Value */);
  }

  // Define a cv::Mat with the output of the Closing
  Mat outputImage(outputRowsBuddyClosing2D, outputColsBuddyClosing2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyClosing2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
