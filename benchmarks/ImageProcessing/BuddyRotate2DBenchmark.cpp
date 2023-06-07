//===- BuddyRotate2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for Rotate2D operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/DIP.h>
#include <buddy/DIP/ImageContainer.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare input image.
Mat inputImageBuddyRotate2D;

// Define the angle.
float BuddyRotate2DAngle;

// Define sizes of input.
intptr_t sizesInputBuddyRotate2D[2];

// Define Angle option selected.
dip::ANGLE_TYPE AngleType;

void initializeBuddyRotate2D(char **argv) {
  inputImageBuddyRotate2D = imread(argv[1], IMREAD_GRAYSCALE);

  sizesInputBuddyRotate2D[0] = inputImageBuddyRotate2D.rows;
  sizesInputBuddyRotate2D[1] = inputImageBuddyRotate2D.cols;

  if (static_cast<string>(argv[2]) == "DEGREE") {
    AngleType = dip::ANGLE_TYPE::DEGREE;
  } else {
    AngleType = dip::ANGLE_TYPE::RADIAN;
  }

  std::string argAngle = argv[3];
  try {
    BuddyRotate2DAngle = std::stof(argAngle);
  } catch (const std::exception &e) {
    cout << "Exception converting rotation angle to float." << endl;
  }
}

static void Buddy_Rotate2D_DEGREE(benchmark::State &state) {
  // Define the MemRef descriptor for input.
  Img<float, 2> inputBuddyRotate2D(inputImageBuddyRotate2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Rotate2D function.
      MemRef<float, 2> output = dip::Rotate2D(
          &inputBuddyRotate2D, BuddyRotate2DAngle, dip::ANGLE_TYPE::DEGREE);
    }
  }
}

static void Buddy_Rotate2D_RADIAN(benchmark::State &state) {
  // Define the MemRef descriptor for input.
  Img<float, 2> inputBuddyRotate2D(inputImageBuddyRotate2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Rotate2D function.
      MemRef<float, 2> output = dip::Rotate2D(
          &inputBuddyRotate2D, BuddyRotate2DAngle, dip::ANGLE_TYPE::RADIAN);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyRotate2D() {
  if (AngleType == dip::ANGLE_TYPE::DEGREE) {
    BENCHMARK(Buddy_Rotate2D_DEGREE)->Arg(1)->Unit(benchmark::kMillisecond);
  } else {
    BENCHMARK(Buddy_Rotate2D_RADIAN)->Arg(1)->Unit(benchmark::kMillisecond);
  }
}

// Generate result image.
void generateResultBuddyRotate2D() {
  // Define the MemRef descriptor for input.
  Img<float, 2> input(inputImageBuddyRotate2D);
  MemRef<float, 2> output(sizesInputBuddyRotate2D);
  // Run the rotate 2D operation.
  if (AngleType == dip::ANGLE_TYPE::DEGREE) {
    // Call the MLIR Rotate2D function.
    output = dip::Rotate2D(&input, BuddyRotate2DAngle, dip::ANGLE_TYPE::DEGREE);
  } else {
    // Call the MLIR Rotate2D function.
    output = dip::Rotate2D(&input, BuddyRotate2DAngle, dip::ANGLE_TYPE::RADIAN);
  }

  // Define a cv::Mat with the output of the rotate operation.
  Mat outputImage(output.getSizes()[0], output.getSizes()[1], CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = true;
  try {
    result = imwrite("ResultBuddyRotate2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
