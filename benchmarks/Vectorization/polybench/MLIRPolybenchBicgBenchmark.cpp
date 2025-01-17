//===- MLIRPolybenchBicgBenchmark.cpp -------------------------------------===//
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
// This file implements the bicg Polybench benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <vector>

extern "C" {
void _mlir_ciface_bicg(int, int, MemRef<double, 2> *, MemRef<double, 1> *,
                       MemRef<double, 1> *, MemRef<double, 1> *,
                       MemRef<double, 1> *);
void _mlir_ciface_bicg_init_array(int, int, MemRef<double, 2> *,
                                  MemRef<double, 1> *, MemRef<double, 1> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {38, 42}},           {"small", {116, 124}},
    {"medium", {390, 410}},       {"large", {1900, 2100}},
    {"extralarge", {1800, 2200}},
};

static void printArray(int m, int n, double *s, double *q) {
  polybench::startDump();
  polybench::beginDump("s");
  for (int i = 0; i < m; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", s[i]);
  }
  polybench::endDump("s");

  polybench::beginDump("q");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", q[i]);
  }
  polybench::endDump("q");
  polybench::finishDump();
}

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({N, M}, 0);
  MemRef<double, 1> inputS({M}, 0);
  MemRef<double, 1> inputQ({N}, 0);
  MemRef<double, 1> inputP({M}, 0);
  MemRef<double, 1> inputR({N}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_bicg_init_array(M, N, &inputA, &inputR, &inputP);
    state.ResumeTiming();
    _mlir_ciface_bicg(M, N, &inputA, &inputS, &inputQ, &inputP, &inputR);
  }
}

void registerMLIRPolybenchBicg(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "bicg-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchBicg(size_t size_id) {
  const std::string benchmarkName = "bicg-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t M = size[0];
  const size_t N = size[1];

  MemRef<double, 2> inputA({N, M}, 0);
  MemRef<double, 1> inputS({M}, 0);
  MemRef<double, 1> inputQ({N}, 0);
  MemRef<double, 1> inputP({M}, 0);
  MemRef<double, 1> inputR({N}, 0);

  _mlir_ciface_bicg_init_array(M, N, &inputA, &inputR, &inputP);

  _mlir_ciface_bicg(M, N, &inputA, &inputS, &inputQ, &inputP, &inputR);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(M, N, inputS.getData(), inputQ.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
