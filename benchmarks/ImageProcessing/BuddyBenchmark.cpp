//===- BuddyBenchmark.cpp -------------------------------------------------===//
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
void _mlir_ciface_conv_2d(MemRef<float, 2> *input, MemRef<float, 2> *kernel,
                          MemRef<float, 2> *output);
}

// Read input image
Mat inputImageBuddy = imread("../../benchmarks/ImageProcessing/Images/YuTu.png",
                             IMREAD_GRAYSCALE);

// Define the kernel.
int kernelRows = laplacianKernelRows;
int kernelCols = laplacianKernelCols;

// Define output for buddy mlir implementation.
int outputRows = inputImageBuddy.rows - kernelRows + 1;
int outputCols = inputImageBuddy.cols - kernelCols + 1;

// Define allocated, sizes, and strides.
intptr_t sizesInput[2] = {inputImageBuddy.rows, inputImageBuddy.cols};
intptr_t sizesKernel[2] = {kernelRows, kernelCols};
intptr_t sizesOutput[2] = {outputRows, outputCols};
intptr_t stridesInput[2] = {inputImageBuddy.rows, inputImageBuddy.cols};
intptr_t stridesKernel[2] = {kernelRows, kernelCols};
intptr_t stridesOutput[2] = {outputRows, outputCols};

// Define input, kernel, and output.
MemRef<float, 2> input(inputImageBuddy, 0, sizesInput, stridesInput);
MemRef<float, 2> kernel(laplacianKernelRows, laplacianKernelCols,
                        laplacianKernelAlign, 0, sizesKernel, stridesKernel);
MemRef<float, 2> output(outputRows, outputCols, 0, sizesOutput, stridesOutput);

static void BM_Buddy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d(&input, &kernel, &output);
    }
  }
}

// Register benchmarking function with different arguments
BENCHMARK(BM_Buddy)->Arg(1);
BENCHMARK(BM_Buddy)->Arg(2);
BENCHMARK(BM_Buddy)->Arg(4);
BENCHMARK(BM_Buddy)->Arg(8);
BENCHMARK(BM_Buddy)->Arg(16);
