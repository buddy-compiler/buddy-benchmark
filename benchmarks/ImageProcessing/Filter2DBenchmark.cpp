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

// Declare input image, kernel and output image.
Mat inputImageFilter2D, kernelFilter2D, outputFilter2D;

void initializeBM_Filter2D_OpenCV(int argc, char **argv) {
  inputImageFilter2D = imread(argv[1], IMREAD_GRAYSCALE);

  kernelFilter2D = Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]),
                       CV_32FC1, get<0>(kernelMap[argv[2]]));
}

// Benchmarking function.
static void BM_Filter2D_OpenCV(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      filter2D(inputImageFilter2D, outputFilter2D, CV_32FC1, kernelFilter2D,
               cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_Filter2D_OpenCV)->Arg(1);
BENCHMARK(BM_Filter2D_OpenCV)->Arg(2);
BENCHMARK(BM_Filter2D_OpenCV)->Arg(4);
BENCHMARK(BM_Filter2D_OpenCV)->Arg(8);
BENCHMARK(BM_Filter2D_OpenCV)->Arg(16);
