//===- BuddyConv2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for buddy-opt tool in buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <buddy/core/Container.h>
#include <buddy/core/ImageContainer.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_buddy_conv_2d(Img<float, 2> *inputBuddyConv2D,
                                MemRef<float, 2> *kernelBuddyConv2D,
                                MemRef<float, 2> *outputBuddyConv2D);
}

// Read input image.
Mat inputImageBuddyConv2D;

// Define the kernel data and size.
float *kernelDataBuddyConv2D;
int kernelRowsBuddyConv2D, kernelColsBuddyConv2D;

// Define the output size.
int outputRowsBuddyConv2D, outputColsBuddyConv2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyConv2D[2];
intptr_t sizesKernelBuddyConv2D[2];
intptr_t sizesOutputBuddyConv2D[2];

void initializeBuddyConv2D(char **argv) {
  inputImageBuddyConv2D = imread(argv[1], IMREAD_GRAYSCALE);

  kernelDataBuddyConv2D = get<0>(kernelMap[argv[2]]);
  kernelRowsBuddyConv2D = get<1>(kernelMap[argv[2]]);
  kernelColsBuddyConv2D = get<2>(kernelMap[argv[2]]);

  outputRowsBuddyConv2D =
      inputImageBuddyConv2D.rows - kernelRowsBuddyConv2D + 1;
  outputColsBuddyConv2D =
      inputImageBuddyConv2D.cols - kernelColsBuddyConv2D + 1;

  sizesInputBuddyConv2D[0] = inputImageBuddyConv2D.rows;
  sizesInputBuddyConv2D[1] = inputImageBuddyConv2D.cols;

  sizesKernelBuddyConv2D[0] = kernelRowsBuddyConv2D;
  sizesKernelBuddyConv2D[1] = kernelColsBuddyConv2D;

  sizesOutputBuddyConv2D[0] = outputRowsBuddyConv2D;
  sizesOutputBuddyConv2D[1] = outputColsBuddyConv2D;
}

static void Buddy_Conv2D(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyConv2D(inputImageBuddyConv2D);
  MemRef<float, 2> kernelBuddyConv2D(kernelDataBuddyConv2D,
                                     sizesKernelBuddyConv2D);
  MemRef<float, 2> outputBuddyConv2D(sizesOutputBuddyConv2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_buddy_conv_2d(&inputBuddyConv2D, &kernelBuddyConv2D,
                                 &outputBuddyConv2D);
    }
  }
}

// Register benchmarking function.
BENCHMARK(Buddy_Conv2D)->Arg(1)->Unit(benchmark::kMillisecond);

// Generate result image.
void generateResultBuddyConv2D(char **argv) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> input(inputImageBuddyConv2D);
  MemRef<float, 2> kernel(kernelDataBuddyConv2D, sizesKernelBuddyConv2D);
  MemRef<float, 2> output(sizesOutputBuddyConv2D);
  // Run the 2D convolution.
  _mlir_ciface_buddy_conv_2d(&input, &kernel, &output);

  // Define a cv::Mat with the output of the convolution.
  Mat outputImage(outputRowsBuddyConv2D, outputColsBuddyConv2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyConv2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
