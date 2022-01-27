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

#include "ImageProcessing/Kernels.h"
#include "Utils/Container.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the conv2d C interface.
extern "C" {
void _mlir_ciface_mlir_conv_2d(MemRef<float, 2> *inputConv2D,
                               MemRef<float, 2> *kernelConv2D,
                               MemRef<float, 2> *outputConv2D);
}

// Read input image.
Mat inputImageMLIRConv2D, kernelMLIRConv2DMat;

// Define the kernel size.
int kernelRowsMLIRConv2D, kernelColsMLIRConv2D;

// Define the output size.
int outputRowsMLIRConv2D, outputColsMLIRConv2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputMLIRConv2D[2];
intptr_t sizesKernelMLIRConv2D[2];
intptr_t sizesOutputMLIRConv2D[2];

void initializeMLIRConv2D(int argc, char **argv) {
  inputImageMLIRConv2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelMLIRConv2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsMLIRConv2D = kernelMLIRConv2DMat.rows;
  kernelColsMLIRConv2D = kernelMLIRConv2DMat.cols;

  outputRowsMLIRConv2D = inputImageMLIRConv2D.rows - kernelRowsMLIRConv2D + 1;
  outputColsMLIRConv2D = inputImageMLIRConv2D.cols - kernelColsMLIRConv2D + 1;

  sizesInputMLIRConv2D[0] = inputImageMLIRConv2D.rows;
  sizesInputMLIRConv2D[1] = inputImageMLIRConv2D.cols;

  sizesKernelMLIRConv2D[0] = kernelRowsMLIRConv2D;
  sizesKernelMLIRConv2D[1] = kernelColsMLIRConv2D;

  sizesOutputMLIRConv2D[0] = outputRowsMLIRConv2D;
  sizesOutputMLIRConv2D[1] = outputColsMLIRConv2D;
}

static void MLIR_Conv2D(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputMLIRConv2D(inputImageMLIRConv2D, sizesInputMLIRConv2D);
  MemRef<float, 2> kernelMLIRConv2D(kernelMLIRConv2DMat, sizesKernelMLIRConv2D);
  MemRef<float, 2> outputMLIRConv2D(sizesOutputMLIRConv2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_conv_2d(&inputMLIRConv2D, &kernelMLIRConv2D,
                                &outputMLIRConv2D);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_Conv2D)->Arg(1);

// Generate result image.
void generateResultMLIRConv2D() {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageMLIRConv2D, sizesInputMLIRConv2D);
  MemRef<float, 2> kernel(kernelMLIRConv2DMat, sizesKernelMLIRConv2D);
  MemRef<float, 2> output(sizesOutputMLIRConv2D);
  // Run the 2D convolution.
  _mlir_ciface_mlir_conv_2d(&input, &kernel, &output);

  // Define a cv::Mat with the output of the convolution.
  Mat outputImage(outputRowsMLIRConv2D, outputColsMLIRConv2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultMLIRConv2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
