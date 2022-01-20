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
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_mobilenet_v3(MemRef<float, 2> *output,
                               MemRef<float, 4> *input);
}

const cv::Mat imagePreprocessing() {

  cv::Mat inputImage = cv::imread(
      "../../benchmarks/DeepLearning/Models/MobileNet-V3/Images/dog.png");
  assert(!inputImage.empty() && "Could not read the image.");
  cv::Mat resizedImage;
  int imageWidth = 224;
  int imageHeight = 224;
  cv::resize(inputImage, resizedImage, cv::Size(imageWidth, imageHeight),
             cv::INTER_LINEAR);
  return resizedImage;
}

cv::Mat image = imagePreprocessing();

intptr_t sizesInput[4] = {1, image.rows, image.cols, 3};
intptr_t sizesOutnput[2] = {1, 1001};

MemRef<float, 4> input(image, sizesInput);
MemRef<float, 2> output(sizesOutnput);

// Define benchmark function.
void BM_MobileNet(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mobilenet_v3(&output, &input);
    }
  }
}

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

std::string getLabel(int idx) {
  std::ifstream in(
      "../../benchmarks/DeepLearning/Models/MobileNet-V3/Labels.txt");
  assert(in.is_open() && "Could not read the label file.");
  std::string label;
  for (int i = 0; i < idx; ++i)
    std::getline(in, label);
  std::getline(in, label);
  in.close();
  return label;
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_MobileNet)->Arg(1);

// Print result function.
void printResult() {
  // Run the model and activation function.
  _mlir_ciface_mobilenet_v3(&output, &input);
  auto out = output.getData();
  softmax(out, 1001);
  // Find the classification and print the result.
  float maxVal = 0;
  float maxIdx = 0;
  for (int i = 0; i < 1001; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification Index: " << maxIdx << std::endl;
  std::cout << "Classification: " << getLabel(maxIdx) << std::endl;
  std::cout << "Probability: " << maxVal << std::endl;
}
