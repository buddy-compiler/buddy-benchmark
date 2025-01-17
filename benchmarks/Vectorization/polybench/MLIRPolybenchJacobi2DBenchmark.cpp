//===- MLIRPolybenchJacobi2DBenchmark.cpp ---------------------------------===//
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
// This file implements the jacobi-2d Polybench benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstdio>
#include <vector>

extern "C" {
void _mlir_ciface_jacobi_2d(int, int, MemRef<double, 2> *, MemRef<double, 2> *);
void _mlir_ciface_jacobi_2d_init_array(int, MemRef<double, 2> *,
                                       MemRef<double, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 30}},           {"small", {40, 90}},
    {"medium", {100, 250}},       {"large", {500, 1300}},
    {"extralarge", {1000, 2800}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({N, N}, 0);
  MemRef<double, 2> inputB({N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_jacobi_2d_init_array(N, &inputA, &inputB);
    state.ResumeTiming();
    _mlir_ciface_jacobi_2d(TSTEPS, N, &inputA, &inputB);
  }
}

static void printArray(int n, double *A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", A[i * n + j]);
    }
  }
  printf("\n");
}

void registerMLIRPolybenchJacobi2D(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "jacobi-2d-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchJacobi2D(size_t size_id) {
  const std::string benchmarkName = "jacobi-2d-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t TSTEPS = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({N, N}, 0);
  MemRef<double, 2> inputB({N, N}, 0);

  _mlir_ciface_jacobi_2d_init_array(N, &inputA, &inputB);
  _mlir_ciface_jacobi_2d(TSTEPS, N, &inputA, &inputB);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputA.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
