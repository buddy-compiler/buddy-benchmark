//===- OpenCVTopHat2DBenchmark.cpp ----------------------------------------===//
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
// This file implements the benchmark for OpenCV TopHat2D.
//
//===----------------------------------------------------------------------===//

#include "ImageProcessing/Kernels.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare input image, kernel and output image.
Mat inputImageTopHat2D, kernelTopHat2D, outputTopHat2D;

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

// Define Boundary option selected.
BoundaryOption OpenCVBoundaryType4;

void initializeOpenCVTopHat2D(char **argv) {
  inputImageTopHat2D = imread(argv[1], IMREAD_GRAYSCALE);

  kernelTopHat2D = Mat(get<1>(kernelMap1[argv[4]]), get<2>(kernelMap1[argv[4]]),
                       CV_8UC1, get<0>(kernelMap1[argv[4]]));

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    OpenCVBoundaryType4 = replicate_padding;
  } else {
    OpenCVBoundaryType4 = constant_padding;
  }
}

// Benchmarking function.
static void OpenCV_TopHat2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
  cv::morphologyEx(inputImageTopHat2D, outputTopHat2D, 5, kernelTopHat2D, cv::Point(1, 1),
               3, cv::BORDER_CONSTANT, 0.0);        
    }
  }
}

static void OpenCV_TopHat2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
  cv::morphologyEx(inputImageTopHat2D, outputTopHat2D, 5, kernelTopHat2D, cv::Point(1, 1),
               3, cv::BORDER_REPLICATE, 0.0); 
    }
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVTopHat2D() {
  if (OpenCVBoundaryType3 == replicate_padding) {
    BENCHMARK(OpenCV_TopHat2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(OpenCV_TopHat2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultOpenCVTopHat2D() {
  if (OpenCVBoundaryType3 == replicate_padding) {
  cv::morphologyEx(inputImageTopHat2D, outputTopHat2D, 5, kernelTopHat2D, cv::Point(1, 1),
               3, cv::BORDER_REPLICATE, 0.0); 
  } else {
  cv::morphologyEx(inputImageTopHat2D, outputTopHat2D, 5, kernelTopHat2D, cv::Point(1, 1),
               3, cv::BORDER_CONSTANT, 0.0); 
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result =
        imwrite("ResultOpenCVTopHat2D.png", outputTopHat2D, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
