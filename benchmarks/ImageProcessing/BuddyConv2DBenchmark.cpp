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
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/imgcodecs/loadsave.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_conv_2d(Img<float, 2> *inputBuddyConv2D,
                          MemRef<float, 2> *kernelBuddyConv2D,
                          MemRef<float, 2> *outputBuddyConv2D);
}

// Name of input image to be read.
std::string inputNameBuddyConv2D;

// Define the kernel data and size.
float *kernelDataBuddyConv2D;
int kernelRowsBuddyConv2D, kernelColsBuddyConv2D;

// Define the output size.
int outputRowsBuddyConv2D, outputColsBuddyConv2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyConv2D[2];
intptr_t sizesKernelBuddyConv2D[2];
intptr_t sizesOutputBuddyConv2D[2];

void initializeBuddyConv2D(char **argv, Img<float, 2> inputImageBuddyConv2D) {
  inputNameBuddyConv2D = argv[1];

  kernelDataBuddyConv2D = get<0>(kernelMap[argv[2]]);
  kernelRowsBuddyConv2D = get<1>(kernelMap[argv[2]]);
  kernelColsBuddyConv2D = get<2>(kernelMap[argv[2]]);

  outputRowsBuddyConv2D =
      inputImageBuddyConv2D.getSizes()[0] - kernelRowsBuddyConv2D + 1;
  outputColsBuddyConv2D =
      inputImageBuddyConv2D.getSizes()[1] - kernelColsBuddyConv2D + 1;

  sizesInputBuddyConv2D[0] = inputImageBuddyConv2D.getSizes()[0];
  sizesInputBuddyConv2D[1] = inputImageBuddyConv2D.getSizes()[1];

  sizesKernelBuddyConv2D[0] = kernelRowsBuddyConv2D;
  sizesKernelBuddyConv2D[1] = kernelColsBuddyConv2D;

  sizesOutputBuddyConv2D[0] = outputRowsBuddyConv2D;
  sizesOutputBuddyConv2D[1] = outputColsBuddyConv2D;
}

static void Buddy_Conv2D(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputBuddyConv2D =
      dip::imread<float, 2>(inputNameBuddyConv2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> kernelBuddyConv2D(kernelDataBuddyConv2D,
                                     sizesKernelBuddyConv2D);
  MemRef<float, 2> outputBuddyConv2D(sizesOutputBuddyConv2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(&inputBuddyConv2D, &kernelBuddyConv2D,
                           &outputBuddyConv2D);
    }
  }
}

// Register benchmarking function.
BENCHMARK(Buddy_Conv2D)->Arg(1)->Unit(benchmark::kMillisecond);

// Generate result image.
void generateResultBuddyConv2D(Img<float, 2> input) {
  // Define the MemRef descriptor for kernel, and output.
  MemRef<float, 2> kernel(kernelDataBuddyConv2D, sizesKernelBuddyConv2D);
  MemRef<float, 2> output(sizesOutputBuddyConv2D);
  // Run the 2D convolution.
  _mlir_ciface_conv_2d(&input, &kernel, &output);

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsBuddyConv2D, outputColsBuddyConv2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultBuddyConv2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
