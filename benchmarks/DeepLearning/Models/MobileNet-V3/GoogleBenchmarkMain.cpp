//===- GoogleBenchmarkMain.cpp --------------------------------------------===//
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
// This file implements the benchmark for Mobilenet-V3 model.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#define INPUT_N 1
#define INPUT_C 3
#define INPUT_H 224
#define INPUT_W 224
#define OUTPUT_N 1000

// Helper functions and variables.
namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

constexpr size_t ParamsSize = 2554968;

bool areArraysEqual(float array1[], float array2[], int size,
                    float epsilon = 0.0001) {
  for (int i = 0; i < size; ++i) {
    if (fabs(array1[i] - array2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

void loadParameters(const std::string &floatParamPath,
                    const std::string &int64ParamPath,
                    MemRef<float, 1> &floatParam,
                    MemRef<long long, 1> &int64Param) {
  std::ifstream floatParamFile(floatParamPath, std::ios::in | std::ios::binary);
  if (!floatParamFile.is_open()) {
    std::string errMsg = "Failed to open float param file: " +
                         std::filesystem::canonical(floatParamPath).string();
    throw std::runtime_error(errMsg);
  }
  floatParamFile.read(reinterpret_cast<char *>(floatParam.getData()),
                      floatParam.getSize() * sizeof(float));
  if (floatParamFile.fail()) {
    throw std::runtime_error("Failed to read float param file");
  }
  floatParamFile.close();

  std::ifstream int64ParamFile(int64ParamPath, std::ios::in | std::ios::binary);
  if (!int64ParamFile.is_open()) {
    std::string errMsg = "Failed to open int64 param file: " +
                         std::filesystem::canonical(int64ParamPath).string();
    throw std::runtime_error(errMsg);
  }
  int64ParamFile.read(reinterpret_cast<char *>(int64Param.getData()),
                      int64Param.getSize() * sizeof(long long));
  if (int64ParamFile.fail()) {
    throw std::runtime_error("Failed to read int64 param file");
  }
  int64ParamFile.close();
}

} // namespace

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_forward(MemRef<float, 2> *output, MemRef<float, 1> *arg0,
                          MemRef<long long, 1> *arg1, Img<float, 4> *input);
}

void BM_MobileNet_V3(benchmark::State &state) {

  // Define the sizes of the input and output tensors.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesOutput[2] = {1, OUTPUT_N};

  // Generate input memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;
  float inputRand[inputSize];

  // Create input and output containers for the image and model output.
  Img<float, 4> input(sizesInput);
  MemRef<float, 2> output(sizesOutput);

  // Load model parameters from the specified file.
  std::string mobilenetDir = getenv("MOBILENETV3_EXAMPLE_PATH");
  std::string paramsDir = mobilenetDir + "/arg0.data";
  std::string intDir = mobilenetDir + "/arg1.data";
  MemRef<float, 1> paramsContainerf32({ParamsSize});
  MemRef<long long, 1> ParamsContainerInt64({34});
  loadParameters(paramsDir, intDir, paramsContainerf32, ParamsContainerInt64);
  for (auto _ : state) {
    _mlir_ciface_forward(&output, &paramsContainerf32, &ParamsContainerInt64,
                         &input);
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_MobileNet_V3)->Unit(benchmark::kMillisecond);

/// Correctness Verification
/// The verification does not affect the performance.
/// - Set the scalar case as the criteria.
/// - Input elements are random numbers.
/// - Output elements are initialized to zero.
/// - Compare the output of various optimizations with the scalar version to
///   verify correctness.
void verification() {
  // Set the random number generator.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  // Define the sizes of the input and output tensors.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};
  intptr_t sizesOutput[2] = {1, OUTPUT_N};

  // Generate input memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }

  // Create input and output containers for the image and model output.
  Img<float, 4> input(inputRand, sizesInput);
  MemRef<float, 2> outputScalar(sizesOutput);
  MemRef<float, 2> outputVectorization(sizesOutput);

  // Load model parameters from the specified file.
  std::string mobilenetDir = getenv("MOBILENETV3_EXAMPLE_PATH");
  std::string paramsDir = mobilenetDir + "/arg0.data";
  std::string intDir = mobilenetDir + "/arg1.data";
  MemRef<float, 1> paramsContainerf32({ParamsSize});
  MemRef<long long, 1> ParamsContainerInt64({34});
  loadParameters(paramsDir, intDir, paramsContainerf32, ParamsContainerInt64);

  // Call the forward function of the model.
  _mlir_ciface_forward(&outputScalar, &paramsContainerf32,
                       &ParamsContainerInt64, &input);
  _mlir_ciface_forward(&outputVectorization, &paramsContainerf32,
                       &ParamsContainerInt64, &input);

  auto resultScalar = outputScalar.getData();
  auto resultVectorization = outputVectorization.getData();

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultVectorization, OUTPUT_N)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}

int main(int argc, char **argv) {
  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  // Run correctness verification.
  verification();
  return 0;
}
