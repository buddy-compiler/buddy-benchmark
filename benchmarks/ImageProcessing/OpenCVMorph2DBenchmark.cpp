//===- OpenCVErode2DBenchmark.cpp ----------------------------------------===//
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
// This file implements the benchmark for OpenCV Erode2D.
//
//===----------------------------------------------------------------------===//

#include "ImageProcessing/Kernels.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare input image, kernel and output image.
Mat inputImageMorph2D, kernelMorph2D, outputErode2D, outputDilate2D,
    outputOpening2D, outputClosing2D, outputTopHat2D, outputBottomHat2D,
    outputMorphGrad2D;

// Declare Boundary Options supported.
enum BoundaryOption { constant_padding, replicate_padding };

BoundaryOption OpenCVBoundaryType1;

void initializeOpenCVMorph2D(char **argv) {
  inputImageMorph2D = imread(argv[1], IMREAD_GRAYSCALE);

  kernelMorph2D = Mat(get<1>(kernelMap1[argv[4]]), get<2>(kernelMap1[argv[4]]),
                      CV_8UC1, get<0>(kernelMap1[argv[4]]));

  if (static_cast<string>(argv[3]) == "REPLICATE_PADDING") {
    OpenCVBoundaryType1 = replicate_padding;
  } else {
    OpenCVBoundaryType1 = constant_padding;
  }
}

// Benchmarking function.
static void OpenCV_Erode2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      erode(inputImageMorph2D, outputErode2D, kernelMorph2D, cv::Point(1, 1), 5,
            cv::BORDER_CONSTANT, 0.0);
    }
  }
}

static void OpenCV_Erode2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      erode(inputImageMorph2D, outputErode2D, kernelMorph2D, cv::Point(1, 1), 5,
            cv::BORDER_REPLICATE, 0.0);
    }
  }
}

// Benchmarking function.
static void OpenCV_Dilate2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dilate(inputImageMorph2D, outputDilate2D, kernelMorph2D, cv::Point(1, 1),
             5, cv::BORDER_CONSTANT, 0.0);
    }
  }
}

static void OpenCV_Dilate2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dilate(inputImageMorph2D, outputDilate2D, kernelMorph2D, cv::Point(1, 1),
             5, cv::BORDER_REPLICATE, 0.0);
    }
  }
}

// Benchmarking function.
static void OpenCV_Opening2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputOpening2D, 2, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
    }
  }
}

static void OpenCV_Opening2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputOpening2D, 2, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
    }
  }
}

// Benchmarking function.
static void OpenCV_Closing2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputClosing2D, 2, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
    }
  }
}

static void OpenCV_Closing2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputClosing2D, 2, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
    }
  }
}

// Benchmarking function.
static void OpenCV_TopHat2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputTopHat2D, 5, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
    }
  }
}

static void OpenCV_TopHat2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputTopHat2D, 5, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
    }
  }
}

// Benchmarking function.
static void OpenCV_BottomHat2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputBottomHat2D, 6, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
    }
  }
}

static void OpenCV_BottomHat2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputBottomHat2D, 6, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
    }
  }
}

// Benchmarking function.
static void OpenCV_MorphGrad2D_Constant_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputMorphGrad2D, 4, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
    }
  }
}

static void OpenCV_MorphGrad2D_Replicate_Padding(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      cv::morphologyEx(inputImageMorph2D, outputMorphGrad2D, 4, kernelMorph2D,
                       cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVErode2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    BENCHMARK(OpenCV_Erode2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(OpenCV_Erode2D_Constant_Padding)->Arg(1);
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVDilate2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    BENCHMARK(OpenCV_Dilate2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(OpenCV_Dilate2D_Constant_Padding)->Arg(1);
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVOpening2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    BENCHMARK(OpenCV_Opening2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(OpenCV_Opening2D_Constant_Padding)->Arg(1);
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVClosing2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    BENCHMARK(OpenCV_Closing2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(OpenCV_Closing2D_Constant_Padding)->Arg(1);
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVTopHat2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    BENCHMARK(OpenCV_TopHat2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(OpenCV_TopHat2D_Constant_Padding)->Arg(1);
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVBottomHat2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    BENCHMARK(OpenCV_BottomHat2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(OpenCV_BottomHat2D_Constant_Padding)->Arg(1);
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVMorphGrad2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    BENCHMARK(OpenCV_MorphGrad2D_Replicate_Padding)->Arg(1);
  } else {
    BENCHMARK(OpenCV_MorphGrad2D_Constant_Padding)->Arg(1);
  }
}

// Generate result image.
void generateResultOpenCVErode2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    erode(inputImageMorph2D, outputErode2D, kernelMorph2D, cv::Point(1, 1), 5,
          cv::BORDER_REPLICATE, 0.0);
  } else {
    erode(inputImageMorph2D, outputErode2D, kernelMorph2D, cv::Point(1, 1), 5,
          cv::BORDER_CONSTANT, 0.0);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result =
        imwrite("ResultOpenCVErode2D.png", outputErode2D, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultOpenCVDilate2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    dilate(inputImageMorph2D, outputDilate2D, kernelMorph2D, cv::Point(1, 1), 5,
           cv::BORDER_REPLICATE, 0.0);
  } else {
    dilate(inputImageMorph2D, outputDilate2D, kernelMorph2D, cv::Point(1, 1), 5,
           cv::BORDER_CONSTANT, 0.0);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result =
        imwrite("ResultOpenCVDilate2D.png", outputDilate2D, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultOpenCVOpening2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    cv::morphologyEx(inputImageMorph2D, outputOpening2D, 2, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
  } else {
    cv::morphologyEx(inputImageMorph2D, outputOpening2D, 2, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultOpenCVOpening2D.png", outputOpening2D,
                     compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultOpenCVClosing2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    cv::morphologyEx(inputImageMorph2D, outputClosing2D, 2, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
  } else {
    cv::morphologyEx(inputImageMorph2D, outputClosing2D, 2, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultOpenCVClosing2D.png", outputClosing2D,
                     compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultOpenCVTopHat2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    cv::morphologyEx(inputImageMorph2D, outputTopHat2D, 5, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
  } else {
    cv::morphologyEx(inputImageMorph2D, outputTopHat2D, 5, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
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

// Generate result image.
void generateResultOpenCVBottomHat2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    cv::morphologyEx(inputImageMorph2D, outputBottomHat2D, 6, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
  } else {
    cv::morphologyEx(inputImageMorph2D, outputBottomHat2D, 6, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultOpenCVBottomHat2D.png", outputBottomHat2D,
                     compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}

// Generate result image.
void generateResultOpenCVMorphGrad2D() {
  if (OpenCVBoundaryType1 == replicate_padding) {
    cv::morphologyEx(inputImageMorph2D, outputMorphGrad2D, 4, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_REPLICATE, 0.0);
  } else {
    cv::morphologyEx(inputImageMorph2D, outputMorphGrad2D, 4, kernelMorph2D,
                     cv::Point(1, 1), 3, cv::BORDER_CONSTANT, 0.0);
  }

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    result = imwrite("ResultOpenCVMorphGrad2D.png", outputMorphGrad2D,
                     compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
