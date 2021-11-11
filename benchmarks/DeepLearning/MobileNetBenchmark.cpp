//===- MobileNetBenchmark.cpp ---------------------------------------------===//
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
// This file implements the benchmark for e2e mobilenet.
//
//===----------------------------------------------------------------------===//

#include "Utils/Container.h"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_mobilenet(MemRef<2> *output, MemRef<4> *input);
}

// TODO: Add input image preprocessing, the current preprocessing only has
// resize step.
Mat imagePreprocessing() {
  Mat inputImage = imread("../../benchmarks/DeepLearning/Images/curtain.png");
  assert(!inputImage.empty() && "Could not read the image.");
  Mat resizedImage;
  int imageWidth = 224;
  int imageHeight = 224;
  cv::resize(inputImage, resizedImage, Size(imageWidth, imageHeight),
             INTER_LINEAR);
  return resizedImage;
}

Mat image = imagePreprocessing();

// TODO: figure out the correct strides layout.
intptr_t sizesInput[4] = {1, image.rows, image.cols, 3};
intptr_t stridesInput[4] = {1, image.rows, image.cols, 3};

intptr_t sizesOutnput[2] = {1, 1001};
intptr_t stridesOutput[2] = {1, 1001};

MemRef<4> input(image, 0, sizesInput, stridesInput);
MemRef<2> output(1001, 0, sizesOutnput, stridesOutput);

// Define benchmark function.
static void BM_MobileNet(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mobilenet(&output, &input);
    }
  }
}

// Register benchmarking function with different arguments.
BENCHMARK(BM_MobileNet)->Arg(1);
BENCHMARK(BM_MobileNet)->Arg(4);

// Softmax function.
void softmax(float *input, size_t size) {
  assert(0 <= size <= sizeof(input) / sizeof(float));
  int i;
  float m, sum, constant;
  m = -INFINITY;
  for (i = 0; i < size; ++i) {
    if (m < input[i]) {
      m = input[i];
    }
  }

  sum = 0.0;
  for (i = 0; i < size; ++i) {
    sum += exp(input[i] - m);
  }

  constant = m + log(sum);
  for (i = 0; i < size; ++i) {
    input[i] = exp(input[i] - constant);
  }
}

// Print result function.
void printResult() {
  // Run the model and activation function.
  _mlir_ciface_mobilenet(&output, &input);
  softmax(output.aligned, 1001);
  // Find the classification and print the result.
  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 1001; ++i) {
    if (output.aligned[i] > maxVal) {
      maxVal = output.aligned[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification Index: " << maxIdx << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;
}
