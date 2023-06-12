//===---------- EfficientNetQuantBenchmark.cpp ----------------------------===//
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
// This file implements the benchmark for efficientnet-quant model.
// `efficientnet.mlir` is generated from the EfficientNet-EdgeTpu(S)-quant model
// in https://coral.ai/models/image-classification/ using iree-import-tflite. In
// order to completely eliminate floating-point operations, the Softmax part is
// taken out of the original model and re-implemented in this cpp file. In
// addition, the `main` function in the mlir file is renamed as `forward` to
// avoid function name conflicts.
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

namespace {

// Declare the efficientnet-quant C interface.
extern "C" {
void _mlir_ciface_forward(MemRef<int8_t, 2> *output, Img<uint8_t, 4> *input);
}

const cv::Mat imagePreprocessing() {
  cv::Mat inputImage = cv::imread(
      "../../benchmarks/DeepLearning/Models/EfficientNet-Quant/Images/"
      "YellowLabradorLooking_new.jpg");
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
intptr_t sizesOutput[2] = {1, 1001};

Img<uint8_t, 4> input(image, sizesInput, false);
MemRef<int8_t, 2> output(sizesOutput);

// Define benchmark function.
void BM_EfficientNet_Quant(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_forward(&output, &input);
    }
  }
}

// Obtained from the original model's pretrained weights.
const float scale_dequant = 8.120160e-02;
const float zero_point_dequant = -5.700000e+01;
const float scale_quant = 2.560000e+02;
const float zero_point_quant = -1.280000e+02;

// Re-implementation of the Softmax part of the original model.
void softmax(int8_t *input, size_t size) {
  assert(0 <= size <= sizeof(input) / sizeof(int8_t));
  float results[1001];
  float m = -INFINITY;
  for (int i = 0; i < size; ++i) {
    // Restore the scaling factor and zero offset used during quantization.
    results[i] =
        (static_cast<float>(input[i]) - zero_point_dequant) * scale_dequant;
    if (m < results[i]) {
      m = results[i];
    }
  }
  float sum = 0.0;
  for (int i = 0; i < size; ++i) {
    results[i] = exp(results[i] - m);
    sum += results[i];
  }
  for (int i = 0; i < size; ++i) {
    results[i] /= sum;
    // Apply quantization using scaling factors and zero offsets.
    results[i] = results[i] * scale_quant + zero_point_quant;
    input[i] = static_cast<float>(results[i]);
  }
}

std::string getLabel(int idx) {
  std::ifstream in(
      "../../benchmarks/DeepLearning/Models/EfficientNet-Quant/Labels.txt");
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
BENCHMARK(BM_EfficientNet_Quant)->Arg(1)->Unit(benchmark::kMillisecond);

// Print result function.
void printResult() {
  // Run the model.
  _mlir_ciface_forward(&output, &input);
  auto out = output.getData();
  softmax(out, 1001);

  // Find the classification and print the result.
  int8_t maxVal = std::numeric_limits<int8_t>::min();
  int maxIdx = 0;
  for (int i = 0; i < 1001; ++i) {
    if (out[i] > maxVal) {
      maxVal = out[i];
      maxIdx = i;
    }
  }
  std::cout << "Classification Index: " << maxIdx << std::endl;
  std::cout << "Classification: " << getLabel(maxIdx) << std::endl;
  std::cout << "Probability(represented in int8 quantization): "
            << static_cast<int>(maxVal) << std::endl;
}
