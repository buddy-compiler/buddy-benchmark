//===- Corr2DBenchmark.cpp ------------------------------------------------===//
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
void _mlir_ciface_corr_2d(MemRef<float, 2> *inputCorr2D,
                          MemRef<float, 2> *kernelCorr2D,
                          MemRef<float, 2> *outputCorr2D, unsigned int centerX,
                          unsigned int centerY, int boundaryOption);
}

// Declare input image and kernel.
Mat inputImageCorr2D, kernelCorr2DMat;

// Define the kernel size.
int kernelRowsCorr2D, kernelColsCorr2D;

// Define the output size.
int outputRowsCorr2D, outputColsCorr2D;

// Define sizes of input, kernel, and output.
intptr_t sizesInputCorr2D[2];
intptr_t sizesKernelCorr2D[2];
intptr_t sizesOutputCorr2D[2];

void initializeBM_Corr2D_Buddy(int argc, char **argv) {
  inputImageCorr2D = imread(argv[1], IMREAD_GRAYSCALE);
  kernelCorr2DMat = Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]),
                        CV_32FC1, get<0>(kernelMap[argv[2]]));

  kernelRowsCorr2D = kernelCorr2DMat.rows;
  kernelColsCorr2D = kernelCorr2DMat.cols;

  outputRowsCorr2D = inputImageCorr2D.rows;
  outputColsCorr2D = inputImageCorr2D.cols;

  sizesInputCorr2D[0] = inputImageCorr2D.rows;
  sizesInputCorr2D[1] = inputImageCorr2D.cols;

  sizesKernelCorr2D[0] = kernelRowsCorr2D;
  sizesKernelCorr2D[1] = kernelColsCorr2D;

  sizesOutputCorr2D[0] = outputRowsCorr2D;
  sizesOutputCorr2D[1] = outputColsCorr2D;
}

static void BM_Corr2D_Buddy(benchmark::State &state) {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> inputCorr2D(inputImageCorr2D, sizesInputCorr2D);
  MemRef<float, 2> kernelCorr2D(kernelCorr2DMat, sizesKernelCorr2D);
  MemRef<float, 2> outputCorr2D(sizesOutputCorr2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_corr_2d(&inputCorr2D, &kernelCorr2D, &outputCorr2D,
                           1 /* Center X */, 1 /* Center Y */,
                           0 /* Boundary Option */);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_Corr2D_Buddy)->Arg(1);
BENCHMARK(BM_Corr2D_Buddy)->Arg(2);
BENCHMARK(BM_Corr2D_Buddy)->Arg(4);
BENCHMARK(BM_Corr2D_Buddy)->Arg(8);
BENCHMARK(BM_Corr2D_Buddy)->Arg(16);

// Generate result image.
void generateResultCorr2D() {
  // Define the MemRef descriptor for input, kernel, and output.
  MemRef<float, 2> input(inputImageCorr2D, sizesInputCorr2D);
  MemRef<float, 2> kernel(kernelCorr2DMat, sizesKernelCorr2D);
  MemRef<float, 2> output(sizesOutputCorr2D);
  // Run the 2D correlation.
  _mlir_ciface_corr_2d(&input, &kernel, &output, 1 /* Center X */,
                       1 /* Center Y */, 0 /* Boundary Option */);

  // Define a cv::Mat with the output of the correlation.
  Mat outputImage(outputRowsCorr2D, outputColsCorr2D, CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultCorr2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
