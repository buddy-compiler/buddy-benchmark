//===- MLIRPolybenchCovarianceBenchmark.cpp -------------------------------===//
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
// This file implements the covariance Polybench benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_covariance(int, int, double, MemRef<double, 2> *,
                             MemRef<double, 2> *, MemRef<double, 1> *);
void _mlir_ciface_covariance_init_array(int, int, MemRef<double, 1> *,
                                        MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {28, 32}},           {"small", {80, 100}},
    {"medium", {240, 260}},       {"large", {1200, 1400}},
    {"extralarge", {2600, 3000}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 1> floatN({1}, 0);
  MemRef<double, 2> inputData({N, M}, 0);
  MemRef<double, 2> inputCov({M, M}, 0);
  MemRef<double, 1> inputMean({M}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_covariance_init_array(M, N, &floatN, &inputData);
    state.ResumeTiming();
    _mlir_ciface_covariance(M, N, floatN.getData()[0], &inputData, &inputCov,
                            &inputMean);
  }
}

static void printArray(int m, double *cov) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", cov[i * m + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchCovariance(
    const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "covariance-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchCovariance(size_t size_id) {
  const std::string benchmarkName = "covariance-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;
  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 1> floatN({1}, 0);
  MemRef<double, 2> inputData({N, M}, 0);
  MemRef<double, 2> inputCov({M, M}, 0);
  MemRef<double, 1> inputMean({M}, 0);

  _mlir_ciface_covariance_init_array(M, N, &floatN, &inputData);

  _mlir_ciface_covariance(M, N, floatN.getData()[0], &inputData, &inputCov,
                          &inputMean);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(M, inputCov.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
