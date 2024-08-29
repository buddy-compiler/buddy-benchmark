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
// This file implements the benchmark for OpenCV Rotate2D.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const float angle = 30.0;

// Declare input image and output image.
Mat inputImageRotate2D, outputImageRotate2D;

// Declare output image size
Size outputImageSizeRotate2D;

Mat CalculateRotationMatrix() {
  // Get the size of the image
  int inputImageHeight = inputImageRotate2D.rows;
  int inputImageWidth = inputImageRotate2D.cols;

  // Calculate the center
  Point2f center(inputImageWidth / 2.0, inputImageHeight / 2.0);

  // Get the rotation matrix
  Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);

  // Get the output image size
  double cos = std::abs(rotationMatrix.at<double>(0, 0));
  double sin = std::abs(rotationMatrix.at<double>(0, 1));
  int outputImageWidth =
      static_cast<int>((inputImageHeight * sin) + (inputImageWidth * cos));
  int outputImageHeight =
      static_cast<int>((inputImageHeight * cos) + (inputImageWidth * sin));

  outputImageSizeRotate2D = Size(outputImageWidth, outputImageHeight);

  //  Adjust the rotation matrix
  rotationMatrix.at<double>(0, 2) += (outputImageWidth / 2.0) - center.x;
  rotationMatrix.at<double>(1, 2) += (outputImageHeight / 2.0) - center.y;

  return rotationMatrix;
}

void initializeOpenCVRotate2D(char **argv) {
  inputImageRotate2D = imread(argv[1], IMREAD_GRAYSCALE);
}

// Benchmarking functions.
static void OpenCV_Rotate2D(benchmark::State &state) {

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      Mat rotationMatrix = CalculateRotationMatrix();
      warpAffine(inputImageRotate2D, outputImageRotate2D, rotationMatrix,
                 outputImageSizeRotate2D);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkOpenCVRotate2D() {
  BENCHMARK(OpenCV_Rotate2D)->Arg(1)->Unit(benchmark::kMillisecond);
}

// Generate result image.
void generateResultOpenCVRotate2D() {

  Mat rotationMatrix = CalculateRotationMatrix();

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = false;
  try {
    warpAffine(inputImageRotate2D, outputImageRotate2D, rotationMatrix,
               outputImageSizeRotate2D);
    result = imwrite("ResultOpenCVRotate2D.png", outputImageRotate2D,
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