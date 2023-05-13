//===- BuddyResize2DBenchmark.cpp -------------------------------------------===//
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
// This file implements the benchmark for Resize2D operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <buddy/DIP/DIP.h>
#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std;

// Declare input image.
Mat inputImageBuddyResize2D;

// Define the output size or factor.
int outputRowsBuddyResize2DLength, outputColsBuddyResize2DLength;
float outputRowsBuddyResize2DFactor, outputColsBuddyResize2DFactor;

// Define sizes of input and output.
intptr_t sizesInputBuddyResize2D[2];
intptr_t sizesOutputBuddyResize2D[2];
std::vector<float> factorsOutputBuddyResize2D = {1.0, 1.0};

// Declare Interpolation option supported.
enum InterpolationOption { bilinear_interpolation, nearest_neighbour_interpolation };

// Declare Scale option supported.
enum ScaleOption { scale_factor, scale_length };

// Define Interpolation option selected.
InterpolationOption InterpolationType;

// Define Scale option selected.
ScaleOption ScaleType;

void initializeBuddyResize2D(char **argv) {
  inputImageBuddyResize2D = imread(argv[1], IMREAD_GRAYSCALE);

  sizesInputBuddyResize2D[0] = inputImageBuddyResize2D.rows;
  sizesInputBuddyResize2D[1] = inputImageBuddyResize2D.cols;

  if (static_cast<string>(argv[2]) == "SCALE_FACTOR") {
    ScaleType = scale_factor;
  } else {
    ScaleType = scale_length;
  }

  std::string argRow = argv[3];
  std::string argCol = argv[4];
  try {
    if (ScaleType == scale_factor) {
      float outputRowsBuddyResize2DFactor = std::stof(argRow);
      float outputColsBuddyResize2DFactor = std::stof(argCol);
      factorsOutputBuddyResize2D[0] = outputRowsBuddyResize2DFactor;
      factorsOutputBuddyResize2D[1] = outputColsBuddyResize2DFactor;
      sizesOutputBuddyResize2D[0] =  sizesInputBuddyResize2D[0] * outputRowsBuddyResize2DFactor;
      sizesOutputBuddyResize2D[1] =  sizesInputBuddyResize2D[1] * outputColsBuddyResize2DFactor;
    } else {
      intptr_t outputRowsBuddyResize2DLength = std::stoi(argRow);
      intptr_t outputColsBuddyResize2DLength = std::stoi(argCol);
      sizesOutputBuddyResize2D[0] = outputRowsBuddyResize2DLength;
      sizesOutputBuddyResize2D[1] = outputColsBuddyResize2DLength;
    }
  } catch (const std::exception& e) {
    cout << "Exception converting row and col scale_factor/scale_length to number." << endl;
  }

  if (static_cast<string>(argv[5]) == "NEAREST_NEIGHBOUR_INTERPOLATION") {
    InterpolationType = nearest_neighbour_interpolation;
  } else {
    InterpolationType = bilinear_interpolation;
  }
}

static void Buddy_Resize2D_Bilinear_Interpolation_Length(benchmark::State &state) {
  // Define the MemRef descriptor for input.
  Img<float, 2> inputBuddyResize2D(inputImageBuddyResize2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Resize2D function.
      MemRef<float, 2> output = dip::Resize2D(&inputBuddyResize2D, 
                                              dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
                                              sizesOutputBuddyResize2D);
    }
  }
}

static void Buddy_Resize2D_Nearest_Neighbour_Interpolation_Length(benchmark::State &state) {
  // Define the MemRef descriptor for input.
  Img<float, 2> inputBuddyResize2D(inputImageBuddyResize2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Resize2D function.
      MemRef<float, 2> output = dip::Resize2D(&inputBuddyResize2D, 
                                              dip::INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION, 
                                              sizesOutputBuddyResize2D);
    }
  }
}

static void Buddy_Resize2D_Bilinear_Interpolation_Factor(benchmark::State &state) {
  // Define the MemRef descriptor for input.
  Img<float, 2> inputBuddyResize2D(inputImageBuddyResize2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Resize2D function.
      MemRef<float, 2> output = dip::Resize2D(&inputBuddyResize2D, 
                                              dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
                                              factorsOutputBuddyResize2D);
    }
  }
}

static void Buddy_Resize2D_Nearest_Neighbour_Interpolation_Factor(benchmark::State &state) {
  // Define the MemRef descriptor for input.
  Img<float, 2> inputBuddyResize2D(inputImageBuddyResize2D);

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // Call the MLIR Resize2D function.
      MemRef<float, 2> output = dip::Resize2D(&inputBuddyResize2D, 
                                              dip::INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION, 
                                              factorsOutputBuddyResize2D);
    }
  }
}

// Register benchmarking function.
void registerBenchmarkBuddyResize2D() {
  if (InterpolationType == nearest_neighbour_interpolation  && ScaleType == scale_factor) {
    BENCHMARK(Buddy_Resize2D_Nearest_Neighbour_Interpolation_Factor)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else if (InterpolationType == bilinear_interpolation  && ScaleType == scale_factor) {
    BENCHMARK(Buddy_Resize2D_Bilinear_Interpolation_Factor)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else if (InterpolationType == nearest_neighbour_interpolation  && ScaleType == scale_length) {
    BENCHMARK(Buddy_Resize2D_Nearest_Neighbour_Interpolation_Length)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  } else if (InterpolationType == bilinear_interpolation  && ScaleType == scale_length) {
    BENCHMARK(Buddy_Resize2D_Bilinear_Interpolation_Length)
        ->Arg(1)
        ->Unit(benchmark::kMillisecond);
  }
}

// Generate result image.
void generateResultBuddyResize2D() {
  // Define the MemRef descriptor for input.
  Img<float, 2> input(inputImageBuddyResize2D);
  MemRef<float, 2> output(sizesOutputBuddyResize2D);
  // Run the resize 2D operation.
  if (InterpolationType == nearest_neighbour_interpolation  && ScaleType == scale_factor) {
    // Call the MLIR Resize2D function.
    output = dip::Resize2D(&input, 
                          dip::INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION, 
                          factorsOutputBuddyResize2D);
  } else if (InterpolationType == bilinear_interpolation  && ScaleType == scale_factor) {
    // Call the MLIR Resize2D function.
    output = dip::Resize2D(&input, 
                          dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
                          factorsOutputBuddyResize2D);
  } else if (InterpolationType == nearest_neighbour_interpolation  && ScaleType == scale_length) {
    // Call the MLIR Resize2D function.
    output = dip::Resize2D(&input, 
                          dip::INTERPOLATION_TYPE::NEAREST_NEIGHBOUR_INTERPOLATION, 
                          sizesOutputBuddyResize2D);
  } else if (InterpolationType == bilinear_interpolation  && ScaleType == scale_length) {
    // Call the MLIR Resize2D function.
    output = dip::Resize2D(&input, 
                          dip::INTERPOLATION_TYPE::BILINEAR_INTERPOLATION,
                          sizesOutputBuddyResize2D);
  }

  // Define a cv::Mat with the output of the resize operation.
  Mat outputImage(output.getSizes()[0], output.getSizes()[1], CV_32FC1,
                  output.getData());

  // Choose a PNG compression level
  vector<int> compressionParams;
  compressionParams.push_back(IMWRITE_PNG_COMPRESSION);
  compressionParams.push_back(9);

  // Write output to PNG.
  bool result = true;
  try {
    result = imwrite("ResultBuddyResize2D.png", outputImage, compressionParams);
  } catch (const cv::Exception &ex) {
    fprintf(stderr, "Exception converting image to PNG format: %s\n",
            ex.what());
  }
  if (result)
    cout << "Saved PNG file." << endl;
  else
    cout << "ERROR: Can't save PNG file." << endl;
}
