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
Mat inputImageConv2D, kernelConv2DMat;

// Define the kernel size.
int kernelRowsConv2D, kernelColsConv2D;

// Define the output size.
int outputRowsConv2D, outputColsConv2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputConv2D[2];
intptr_t sizesKernelConv2D[2];
intptr_t sizesOutputConv2D[2];

void initializeBM_Conv2D_Buddy(int argc, char **argv) {
  inputImageConv2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelConv2DMat = Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]),
                        CV_32FC1, get<0>(kernelMap[argv[2]]));

  kernelRowsConv2D = kernelConv2DMat.rows;
  kernelColsConv2D = kernelConv2DMat.cols;

  outputRowsConv2D = inputImageConv2D.rows - kernelRowsConv2D + 1;
  outputColsConv2D = inputImageConv2D.cols - kernelColsConv2D + 1;

  sizesInputConv2D[0] = inputImageConv2D.rows;
  sizesInputConv2D[1] = inputImageConv2D.cols;

  sizesKernelConv2D[0] = kernelRowsConv2D;
  sizesKernelConv2D[1] = kernelColsConv2D;

  sizesOutputConv2D[0] = outputRowsConv2D;
  sizesOutputConv2D[1] = outputColsConv2D;
}

static void BM_Conv2D_Buddy(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputConv2D(inputImageConv2D, sizesInputConv2D);
  MemRef<float, 2> kernelConv2D(kernelConv2DMat, sizesKernelConv2D);
  MemRef<float, 2> outputConv2D(sizesOutputConv2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(&inputConv2D, &kernelConv2D, &outputConv2D);
    }
  }
}

// Register benchmarking function.
BENCHMARK(BM_Conv2D_Buddy)->Arg(1);

// Generate result image.
void generateResultConv2D() {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageConv2D, sizesInputConv2D);
  MemRef<float, 2> kernel(kernelConv2DMat, sizesKernelConv2D);
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
