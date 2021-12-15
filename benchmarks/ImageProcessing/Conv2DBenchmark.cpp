//===- Conv2DBenchmark.cpp ------------------------------------------------===//
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
void _mlir_ciface_conv_2d(MemRef<float, 2> *inputConv2D,
                          MemRef<float, 2> *kernelConv2D,
                          MemRef<float, 2> *outputConv2D);
}

// Read input image.
Mat inputImageConv2D = imread(
    "../../benchmarks/ImageProcessing/Images/YuTu.png", IMREAD_GRAYSCALE);

// Define the kernel size.
int kernelRowsConv2D = laplacianKernelRows;
int kernelColsConv2D = laplacianKernelCols;

// Define the output size.
int outputRowsConv2D = inputImageConv2D.rows - kernelRowsConv2D + 1;
int outputColsConv2D = inputImageConv2D.cols - kernelColsConv2D + 1;

// Define sizes of input, kernel, and output.
intptr_t sizesInputConv2D[2] = {inputImageConv2D.rows, inputImageConv2D.cols};
intptr_t sizesKernelConv2D[2] = {kernelRowsConv2D, kernelColsConv2D};
intptr_t sizesOutputConv2D[2] = {outputRowsConv2D, outputColsConv2D};

// Define the MemRef descriptor for input, kernel, and output.
MemRef<float, 2> inputConv2D(inputImageConv2D, sizesInputConv2D);
MemRef<float, 2> kernelConv2D(laplacianKernelAlign, sizesKernelConv2D);
MemRef<float, 2> outputConv2D(sizesOutputConv2D);

static void BM_Conv2D_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(&inputConv2D, &kernelConv2D, &outputConv2D);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_Conv2D_Buddy)->Arg(1);
BENCHMARK(BM_Conv2D_Buddy)->Arg(2);
BENCHMARK(BM_Conv2D_Buddy)->Arg(4);
BENCHMARK(BM_Conv2D_Buddy)->Arg(8);
BENCHMARK(BM_Conv2D_Buddy)->Arg(16);

// Generate result image.
void generateResultConv2D() {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageConv2D, sizesInputConv2D);
  MemRef<float, 2> kernel(laplacianKernelAlign, sizesKernelConv2D);
  MemRef<float, 2> output(sizesOutputConv2D);
  // Run the 2D convolution.
  _mlir_ciface_conv_2d(&input, &kernel, &output);

  // Define a cv::Mat with the output of the convolution.
  Mat outputImage(outputRowsConv2D, outputColsConv2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultConv2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
