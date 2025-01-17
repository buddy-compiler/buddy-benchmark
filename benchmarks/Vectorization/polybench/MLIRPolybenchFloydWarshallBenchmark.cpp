//===- MLIRPolybenchFloydWarshall.cpp -------------------------------------===//
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
// This file implements the floyd-warshall Polybench benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_floyd_warshall(int, MemRef<int, 2> *);
void _mlir_ciface_floyd_warshall_init_array(int, MemRef<int, 2> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {60}},    {"small", {180}},       {"medium", {500}},
    {"large", {2800}}, {"extralarge", {5600}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t N = size[0];

  MemRef<int, 2> inputPath({N, N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_floyd_warshall_init_array(N, &inputPath);
    state.ResumeTiming();
    _mlir_ciface_floyd_warshall(N, &inputPath);
  }
}

static void printArray(int n, int *path) {
  int i, j;
  polybench::startDump();
  polybench::beginDump("path");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%d ", path[i * n + j]);
    }
  }
  polybench::endDump("path");
  polybench::finishDump();
}

void registerMLIRPolybenchFloydWarshall(
    const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "floyd-warshall-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchFloydWarshall(size_t size_id) {
  const std::string benchmarkName = "floyd-warshall-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t N = size[0];

  MemRef<int, 2> inputPath({N, N}, 0);

  _mlir_ciface_floyd_warshall_init_array(N, &inputPath);
  _mlir_ciface_floyd_warshall(N, &inputPath);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(N, inputPath.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
