//===- OpenCVRotate2DBenchmark.cpp ----------------------------------------===//
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
// This file implements the benchmark for OpenCV's Rotate Operations.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare input and output image.
Mat inputImageOpenCVRotate2D, outputImageOpenCVRotate2D;

// Define the angle.
int OpenCVRotate2DAngle;

// Define sizes of input.
intptr_t sizesInputOpenCVRotate2D[2];
cv::Size sizesOutputOpenCVRotate2D;

// Declare Angle option supported.
enum AngleOption { ANGLE_DEGREE, ANGLE_RADIAN };

// Define Angle option selected.
AngleOption OpenCVAngleType;

// Define OpenCV Rotate option.
cv::RotateFlags RotateFlag = cv::ROTATE_90_CLOCKWISE;

// Define the OpenCV Rotate benchmark option.
bool OpenCVRunRotate = true;

void initializeOpenCVRotate2D(char **argv) {
  inputImageOpenCVRotate2D = imread(argv[1], IMREAD_GRAYSCALE);

  sizesInputOpenCVRotate2D[0] = inputImageOpenCVRotate2D.rows;
  sizesInputOpenCVRotate2D[1] = inputImageOpenCVRotate2D.cols;

  if (static_cast<string>(argv[2]) == "DEGREE") {
    OpenCVAngleType = ANGLE_DEGREE;
  } else {
    OpenCVAngleType = ANGLE_RADIAN;
  }

  std::string argAngle = argv[3];
  try {
    OpenCVRotate2DAngle = std::stoi(argAngle);
    OpenCVRotate2DAngle = OpenCVRotate2DAngle % 360;
  } catch (const std::exception& e) {
    cout << "OpenCV rotate() support three ways: 90 degrees clockwise, 180 degrees clockwise, 270 degrees clockwise." << endl;
  }
  if (OpenCVRotate2DAngle == 90) {
    RotateFlag = cv::ROTATE_90_CLOCKWISE;
  } else if (OpenCVRotate2DAngle == 180) {
    RotateFlag = cv::ROTATE_180;
  } else if (OpenCVRotate2DAngle == 270) {
    RotateFlag = cv::ROTATE_90_COUNTERCLOCKWISE;
  } else {
    OpenCVRunRotate = false;
  }
}

// Benchmarking function.
static void OpenCV_Rotate2D_ANGLE_DEGREE(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
        cv::rotate(inputImageOpenCVRotate2D, outputImageOpenCVRotate2D, RotateFlag);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVRotate2D() {
  if (OpenCVAngleType == ANGLE_DEGREE && OpenCVRunRotate == true) {
    BENCHMARK(OpenCV_Rotate2D_ANGLE_DEGREE)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } 
}

// Generate result image.
void generateResultOpenCVRotate2D() {
  // Run the resize 2D operation.
  if (OpenCVAngleType == ANGLE_DEGREE && OpenCVRunRotate == true) {
    cv::rotate(inputImageOpenCVRotate2D, outputImageOpenCVRotate2D, OpenCVRotate2DAngle);

    // Choose a PNG compression level
    vector<int> compressionParams;
    compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
    compressionParams.push_back(9);

    // Write output to PNG.
    bool result = false;
    try {
        result =
            imwrite("ResultOpenCVRotate2D.png", outputImageOpenCVRotate2D, compressionParams);
    } catch (const cv::Exception &ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n",
                ex.what());
    }
    if (result)
        cout << "Saved PNG file." << endl;
    else
        cout << "ERROR: Can't save PNG file." << endl;
  } 
}
