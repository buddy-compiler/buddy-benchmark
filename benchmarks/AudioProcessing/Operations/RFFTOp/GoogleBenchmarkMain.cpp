//===- GoogleBenchmarkMain.cpp---------------------------------------------===//
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
// This file implements the benchmark for RFFT operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DAP/DAP.h>
#include <iostream>
#include <random>

#define testLength 20

// Helper functions and variables.
namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(float array1[], float array2[], int size) {
  for (int i = 0; i < size; ++i) {
    if (array1[i] != array2[i]) {
      return false;
    }
  }
  return true;
}
double *inputAlign0 = new double[testLength];
intptr_t inputSizes0[1] = {testLength};

void initializeInputAlign0() {

  for (int i = 0; i < testLength; ++i) {
    inputAlign0[i] = static_cast<double>(i);
  }
}

} // namespace

static void BUDDY_RFFT(benchmark::State &state) {
  MemRef<double, 1> inputMemRef0(inputAlign0, inputSizes0);
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dap::RFFT(&inputMemRef0);
    }
  }
}

BENCHMARK(BUDDY_RFFT)->Arg(1)->Unit(benchmark::kMillisecond);

void verification() {

  std::vector<double> fileData;
  std::ifstream inputFile(
      "../../benchmarks/DeepLearning/Ops/RFFTOp/result.txt");

  double value;
  while (inputFile >> value) {
    fileData.push_back(value);
  }
  inputFile.close();

  double *inputAlign = new double[testLength];
  for (int i = 0; i < testLength; ++i) {
    inputAlign[i] = static_cast<double>(i);
  }
  intptr_t inputSizes[1] = {testLength};
  MemRef<double, 1> inputMemRef(inputAlign, inputSizes);

  dap::RFFT(&inputMemRef);

  // Get the result array.
  auto resultRFFT = inputMemRef.getData();

  std::cout << "Length : " << fileData.size() << std::endl;

  bool isEqual = true;
  double tolerance = 1e-2;
  size_t minSize = fileData.size();
  for (size_t i = 0; i < minSize; ++i) {
    if (std::abs(resultRFFT[i] - fileData[i]) > tolerance) {
      isEqual = false;
    }
  }

  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification: " << (isEqual ? PASS : FAIL)
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;

}

int main(int argc, char **argv) {
  // Run benchmark.
  initializeInputAlign0();

  ::benchmark::Initialize(&argc, argv);

  ::benchmark::RunSpecifiedBenchmarks();

  // Run correctness verification.
  verification();
  return 0;
}
