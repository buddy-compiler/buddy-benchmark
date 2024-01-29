//===- MLIRConv2DBenchmark.cpp --------------------------------------------===//
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
void _mlir_ciface_mlir_conv_2d(Img<float, 2> *inputConv2D,
                               MemRef<float, 2> *kernelConv2D,
                               MemRef<float, 2> *outputConv2D);
}

// Name of input image to be read.
std::string inputNameMLIRConv2D;

// Define the kernel data and kernel size.
float *kernelDataMLIRConv2D;
int kernelRowsMLIRConv2D, kernelColsMLIRConv2D;

// Define the output size.
int outputRowsMLIRConv2D, outputColsMLIRConv2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputMLIRConv2D[2];
intptr_t sizesKernelMLIRConv2D[2];
intptr_t sizesOutputMLIRConv2D[2];

void initializeMLIRConv2D(char **argv, Img<float, 2> inputImageMLIRConv2D) {
  inputNameMLIRConv2D = argv[1];

  kernelDataMLIRConv2D = get<0>(kernelMap[argv[2]]);
  kernelRowsMLIRConv2D = get<1>(kernelMap[argv[2]]);
  kernelColsMLIRConv2D = get<2>(kernelMap[argv[2]]);

  outputRowsMLIRConv2D =
      inputImageMLIRConv2D.getSizes()[0] - kernelRowsMLIRConv2D + 1;
  outputColsMLIRConv2D =
      inputImageMLIRConv2D.getSizes()[1] - kernelColsMLIRConv2D + 1;

  sizesInputMLIRConv2D[0] = inputImageMLIRConv2D.getSizes()[0];
  sizesInputMLIRConv2D[1] = inputImageMLIRConv2D.getSizes()[1];

  sizesKernelMLIRConv2D[0] = kernelRowsMLIRConv2D;
  sizesKernelMLIRConv2D[1] = kernelColsMLIRConv2D;

  sizesOutputMLIRConv2D[0] = outputRowsMLIRConv2D;
  sizesOutputMLIRConv2D[1] = outputColsMLIRConv2D;
}

static void MLIR_Conv2D(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  Img<float, 2> inputMLIRConv2D =
      dip::imread<float, 2>(inputNameMLIRConv2D, dip::IMGRD_GRAYSCALE);
  MemRef<float, 2> kernelMLIRConv2D(kernelDataMLIRConv2D,
                                    sizesKernelMLIRConv2D);
  MemRef<float, 2> outputMLIRConv2D(sizesOutputMLIRConv2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_conv_2d(&inputMLIRConv2D, &kernelMLIRConv2D,
                                &outputMLIRConv2D);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_Conv2D)->Arg(1)->Unit(benchmark::kMillisecond);

// Generate result image.
void generateResultMLIRConv2D(Img<float, 2> input) {
  // Define the MemRef descriptor for kernel, and output.
  MemRef<float, 2> kernel(kernelDataMLIRConv2D, sizesKernelMLIRConv2D);
  MemRef<float, 2> output(sizesOutputMLIRConv2D);
  // Run the 2D convolution.
  _mlir_ciface_mlir_conv_2d(&input, &kernel, &output);

  // Define an Img container for the output image.
  intptr_t outputSizes[2] = {outputRowsMLIRConv2D, outputColsMLIRConv2D};
  Img<float, 2> outputImage(output.getData(), outputSizes);

  // Write output to PNG.
  bool result = dip::imwrite("ResultMLIRConv2D.png", outputImage);

  if (!result) {
    fprintf(stderr, "Exception converting image to PNG format. \n");
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
