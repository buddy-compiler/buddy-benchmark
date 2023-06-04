//===- OpenCVResize2DBenchmark.cpp ----------------------------------------===//
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
// This file implements the benchmark for OpenCV's Resize Operations.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare input image and output image.
Mat inputImageOpenCVResize2D,  outputImageOpenCVResize2D;

// Define the output size or factor.
int outputRowsOpenCVResize2DLength, outputColsOpenCVResize2DLength;
float outputRowsOpenCVResize2DFactor, outputColsOpenCVResize2DFactor;

// Define sizes of input and output.
intptr_t sizesInputOpenCVResize2D[2];
cv::Size sizesOutputOpenCVResize2D;
std::vector<float> factorsOutputOpenCVResize2D = {1.0, 1.0};

// Declare Interpolation option supported.
enum InterpolationOption { bilinear_interpolation, nearest_neighbour_interpolation };

// Declare Scale option supported.
enum ScaleOption { scale_factor, scale_length };

// Define Interpolation option selected.
InterpolationOption OpenCVInterpolationType;

// Define Scale option selected.
ScaleOption OpenCVScaleType;

void initializeOpenCVResize2D(char **argv) {
  inputImageOpenCVResize2D = imread(argv[1], IMREAD_GRAYSCALE);

  sizesInputOpenCVResize2D[0] = inputImageOpenCVResize2D.rows;
  sizesInputOpenCVResize2D[1] = inputImageOpenCVResize2D.cols;

  if (static_cast<string>(argv[2]) == "SCALE_FACTOR") {
    OpenCVScaleType = scale_factor;
  } else {
    OpenCVScaleType = scale_length;
  }
  
  // Adjust to OpenCV [Col, Row] format.
  std::string argRow = argv[3];
  std::string argCol = argv[4];
  try {
    if (OpenCVScaleType == scale_factor) {
      float outputRowsOpenCVResize2DFactor = std::stof(argRow);
      float outputColsOpenCVResize2DFactor = std::stof(argCol);
      factorsOutputOpenCVResize2D[0] = outputColsOpenCVResize2DFactor;
      factorsOutputOpenCVResize2D[1] = outputRowsOpenCVResize2DFactor;
    } else {
      outputRowsOpenCVResize2DLength = std::stoi(argRow);
      outputColsOpenCVResize2DLength = std::stoi(argCol);
      sizesOutputOpenCVResize2D= cv::Size(outputColsOpenCVResize2DLength, outputRowsOpenCVResize2DLength);
    }
  } catch (const std::exception& e) {
    cout << "Exception converting row and col scale_factor/scale_length to number." << endl;
  }

  if (static_cast<string>(argv[5]) == "NEAREST_NEIGHBOUR_INTERPOLATION") {
    OpenCVInterpolationType = nearest_neighbour_interpolation;
  } else {
    OpenCVInterpolationType = bilinear_interpolation;
  }
}

// Benchmarking function.
static void OpenCV_Resize2D_Bilinear_Interpolation_Length(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
        cv::resize(inputImageOpenCVResize2D, outputImageOpenCVResize2D, sizesOutputOpenCVResize2D, 
                   0, 0, cv::INTER_LINEAR);
    }
  }
}

static void OpenCV_Resize2D_Nearest_Neighbour_Interpolation_Length(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::resize(inputImageOpenCVResize2D, outputImageOpenCVResize2D, sizesOutputOpenCVResize2D, 
                 0, 0, cv::INTER_NEAREST);
    }
  }
}

static void OpenCV_Resize2D_Bilinear_Interpolation_Factor(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::resize(inputImageOpenCVResize2D, outputImageOpenCVResize2D, cv::Size(0, 0), 
                 factorsOutputOpenCVResize2D[0], factorsOutputOpenCVResize2D[1], cv::INTER_LINEAR);
    }
  }
}

static void OpenCV_Resize2D_Nearest_Neighbour_Interpolation_Factor(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::resize(inputImageOpenCVResize2D, outputImageOpenCVResize2D, cv::Size(0, 0), 
                 factorsOutputOpenCVResize2D[0], factorsOutputOpenCVResize2D[1], cv::INTER_NEAREST);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVResize2D() {
  if (OpenCVInterpolationType == nearest_neighbour_interpolation  && OpenCVScaleType == scale_factor) {
    BENCHMARK(OpenCV_Resize2D_Nearest_Neighbour_Interpolation_Factor)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else if (OpenCVInterpolationType == bilinear_interpolation  && OpenCVScaleType == scale_factor) {
    BENCHMARK(OpenCV_Resize2D_Bilinear_Interpolation_Factor)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else if (OpenCVInterpolationType == nearest_neighbour_interpolation  && OpenCVScaleType == scale_length) {
    BENCHMARK(OpenCV_Resize2D_Nearest_Neighbour_Interpolation_Length)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else if (OpenCVInterpolationType == bilinear_interpolation  && OpenCVScaleType == scale_length) {
    BENCHMARK(OpenCV_Resize2D_Bilinear_Interpolation_Length)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Generate result image.
void generateResultOpenCVResize2D() {
  // Run the resize 2D operation.
  if (OpenCVInterpolationType == nearest_neighbour_interpolation  && OpenCVScaleType == scale_factor) {
    cv::resize(inputImageOpenCVResize2D, outputImageOpenCVResize2D, cv::Size(0, 0), 
               factorsOutputOpenCVResize2D[0], factorsOutputOpenCVResize2D[1], cv::INTER_NEAREST);
  } else if (OpenCVInterpolationType == bilinear_interpolation  && OpenCVScaleType == scale_factor) {
    cv::resize(inputImageOpenCVResize2D, outputImageOpenCVResize2D, cv::Size(0, 0), 
               factorsOutputOpenCVResize2D[0], factorsOutputOpenCVResize2D[1], cv::INTER_LINEAR);
  } else if (OpenCVInterpolationType == nearest_neighbour_interpolation  && OpenCVScaleType == scale_length) {
    cv::resize(inputImageOpenCVResize2D, outputImageOpenCVResize2D, sizesOutputOpenCVResize2D, 
               0, 0, cv::INTER_NEAREST);
  } else if (OpenCVInterpolationType == bilinear_interpolation  && OpenCVScaleType == scale_length) {
    cv::resize(inputImageOpenCVResize2D, outputImageOpenCVResize2D, sizesOutputOpenCVResize2D, 
               0, 0, cv::INTER_LINEAR);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result =
        imwrite("ResultOpenCVResize2D.png", outputImageOpenCVResize2D, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
