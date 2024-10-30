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

#define testLength 840

// Helper functions and variables.
namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

double *inputData = new double[testLength];
intptr_t inputSize[1] = {testLength};

void initializeBuddyRFFT() {
  for (int i = 0; i < testLength; ++i) {
    inputData[i] = static_cast<double>(i);
  }
}

MemRef<double, 1> inputMemRef0(inputData, inputSize);
} // namespace

static void BUDDY_RFFT(benchmark::State &state) {
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
      "../../benchmarks/AudioProcessing/Operations/RFFTOp/result.txt");

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

  bool isEqual = true;
  double tolerance = 1e-2;
  size_t minSize = fileData.size();
  for (size_t i = 0; i < minSize; ++i) {
    if (std::abs(resultRFFT[i] - fileData[i]) > tolerance) {
      isEqual = false;
    }
  }

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  std::cout << "Test Length : \033[32m" << fileData.size() << "\033[0m"
            << std::endl;
  std::cout << "Correctness Verification: " << (isEqual ? PASS : FAIL)
            << std::endl;
}

int main(int argc, char **argv) {
  // Run benchmark.
  initializeBuddyRFFT();

  ::benchmark::Initialize(&argc, argv);

  ::benchmark::RunSpecifiedBenchmarks();

  // Run correctness verification.
  verification();
  return 0;
}
