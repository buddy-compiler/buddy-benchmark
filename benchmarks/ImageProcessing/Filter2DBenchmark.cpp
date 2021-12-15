//===- Filter2DBenchmark.cpp ----------------------------------------------===//
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
// This file implements the benchmark for OpenCV filter2D.
//
//===----------------------------------------------------------------------===//

#include "ImageProcessing/Kernels.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Read input image and specify kernel.
Mat inputImageFilter2D = imread(
    "../../benchmarks/ImageProcessing/Images/YuTu.png", IMREAD_GRAYSCALE);
Mat kernelFilter2D = Mat(3, 3, CV_32FC1, laplacianKernelAlign);

// Declare output image.
Mat outputFilter2D;

// Benchmarking function.
static void BM_Filter2D_OpenCV(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      filter2D(inputImageFilter2D, outputFilter2D, CV_32FC1, kernelFilter2D);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_Filter2D_OpenCV)->Arg(1);
BENCHMARK(BM_Filter2D_OpenCV)->Arg(2);
BENCHMARK(BM_Filter2D_OpenCV)->Arg(4);
BENCHMARK(BM_Filter2D_OpenCV)->Arg(8);
BENCHMARK(BM_Filter2D_OpenCV)->Arg(16);
