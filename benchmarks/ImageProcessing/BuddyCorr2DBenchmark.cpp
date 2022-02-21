//===- BuddyCorr2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for Corr2D operation.
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
void _mlir_ciface_corr_2d(MemRef<float, 2> *inputBuddyCorr2D,
                          MemRef<float, 2> *kernelBuddyCorr2D,
                          MemRef<float, 2> *outputBuddyCorr2D,
                          unsigned int centerX, unsigned int centerY,
                          int boundaryOption);
}

// Declare input image and kernel.
Mat inputImageBuddyCorr2D, kernelBuddyCorr2DMat;

// Define the kernel size.
int kernelRowsBuddyCorr2D, kernelColsBuddyCorr2D;

// Define the output size.
int outputRowsBuddyCorr2D, outputColsBuddyCorr2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputBuddyCorr2D[2];
intptr_t sizesKernelBuddyCorr2D[2];
intptr_t sizesOutputBuddyCorr2D[2];

void initializeBuddyCorr2D(int argc, char **argv) {
  inputImageBuddyCorr2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelBuddyCorr2DMat =
      Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]), CV_32FC1,
          get<0>(kernelMap[argv[2]]));

  kernelRowsBuddyCorr2D = kernelBuddyCorr2DMat.rows;
  kernelColsBuddyCorr2D = kernelBuddyCorr2DMat.cols;

  outputRowsBuddyCorr2D = inputImageBuddyCorr2D.rows;
  outputColsBuddyCorr2D = inputImageBuddyCorr2D.cols;

  sizesInputBuddyCorr2D[0] = inputImageBuddyCorr2D.rows;
  sizesInputBuddyCorr2D[1] = inputImageBuddyCorr2D.cols;

  sizesKernelBuddyCorr2D[0] = kernelRowsBuddyCorr2D;
  sizesKernelBuddyCorr2D[1] = kernelColsBuddyCorr2D;

  sizesOutputBuddyCorr2D[0] = outputRowsBuddyCorr2D;
  sizesOutputBuddyCorr2D[1] = outputColsBuddyCorr2D;
}

static void Buddy_Corr2D(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputBuddyCorr2D(inputImageBuddyCorr2D,
                                    sizesInputBuddyCorr2D);
  MemRef<float, 2> kernelBuddyCorr2D(kernelBuddyCorr2DMat,
                                     sizesKernelBuddyCorr2D);
  MemRef<float, 2> outputBuddyCorr2D(sizesOutputBuddyCorr2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_corr_2d(&inputBuddyCorr2D, &kernelBuddyCorr2D,
                           &outputBuddyCorr2D, 1 /* Center X */,
                           1 /* Center Y */, 0 /* Boundary Option */);
    }
  }
}

// Register benchmarking function.
BENCHMARK(Buddy_Corr2D)->Arg(1);

// Generate result image.
void generateResultBuddyCorr2D() {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageBuddyCorr2D, sizesInputBuddyCorr2D);
  MemRef<float, 2> kernel(kernelBuddyCorr2DMat, sizesKernelBuddyCorr2D);
  MemRef<float, 2> output(sizesOutputBuddyCorr2D);
  // Run the 2D correlation.
  _mlir_ciface_corr_2d(&input, &kernel, &output, 1 /* Center X */,
                       1 /* Center Y */, 0 /* Boundary Option */);

  // Define a cv::Mat with the output of the correlation.
  Mat outputImage(outputRowsBuddyCorr2D, outputColsBuddyCorr2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultBuddyCorr2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
