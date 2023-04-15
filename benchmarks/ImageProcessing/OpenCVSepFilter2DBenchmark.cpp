//===- OpenCVSepFilter2DBenchmark.cpp ----------------------------------------===//
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
// This file implements the benchmark for OpenCV sepfilter2D.
//
//===----------------------------------------------------------------------===//


#include "Kernels.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare input image, kernel and output image.
Mat inputImageFilter2D, kernelFilter2DX, kernelFilter2DY, outputFilter2D;

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption OpenCVBoundaryType;

void initializeOpenCVFilter2D(char **argv) {
  inputImageFilter2D = imread(argv[1], IMREAD_GRAYSCALE);

  kernelFilter2DX = Mat(get<1>(kernelMap[argv[2]]), get<2>(kernelMap[argv[2]]),
                       CV_32FC1, get<0>(kernelMap[argv[2]]));
  kernelFilter2DY = Mat(get<1>(kernelMap[argv[3]]), get<2>(kernelMap[argv[3]]),
                       CV_32FC1, get<0>(kernelMap[argv[3]]));                     

  if (static_cast<string>(argv[4]) == "REPLICATE_PADDING") {
    OpenCVBoundaryType = replicate_padding;
  } else {
    OpenCVBoundaryType = constant_padding;
  }
}

// Benchmarking function.
static void OpenCV_SepFilter2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      sepFilter2D(inputImageFilter2D, outputFilter2D, CV_32FC1, kernelFilter2DX, kernelFilter2DY,
               cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
    }
  }
}

static void OpenCV_SepFilter2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      sepFilter2D(inputImageFilter2D, outputFilter2D, CV_32FC1, kernelFilter2DX, kernelFilter2DY,
               cv::Point(0, 0), 0.0, cv::BORDER_REPLICATE);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVFilter2D() {
  if (OpenCVBoundaryType == replicate_padding) {
    BENCHMARK(OpenCV_SepFilter2D_Replicate_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(OpenCV_SepFilter2D_Constant_Padding)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Generate result image.
void generateResultOpenCVSepFilter2D() {
  if (OpenCVBoundaryType == replicate_padding) {
    sepFilter2D(inputImageFilter2D, outputFilter2D, CV_32FC1, kernelFilter2DX, kernelFilter2DY,
             cv::Point(0, 0), 0.0, cv::BORDER_REPLICATE);
  } else {
    sepFilter2D(inputImageFilter2D, outputFilter2D, CV_32FC1, kernelFilter2DX, kernelFilter2DY,
             cv::Point(0, 0), 0.0, cv::BORDER_CONSTANT);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result =
        imwrite("ResultOpenCVFilter2D.png", outputFilter2D, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}