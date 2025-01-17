//===- MLIRPolybenchFdtd2DBenchmark.cpp -----------------------------------===//
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
// This file implements the fdtd-2d Polybench benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cstdio>
#include <vector>

extern "C" {
void _mlir_ciface_fdtd_2d(int, int, int, MemRef<double, 2> *,
                          MemRef<double, 2> *, MemRef<double, 2> *,
                          MemRef<double, 1> *);
void _mlir_ciface_fdtd_2d_init_array(int, int, int, MemRef<double, 2> *,
                                     MemRef<double, 2> *, MemRef<double, 2> *,
                                     MemRef<double, 1> *);
}

const std::vector<std::pair<std::string, std::vector<size_t>>> sizes = {
    {"mini", {20, 20, 30}},
    {"small", {40, 60, 80}},
    {"medium", {100, 200, 240}},
    {"large", {500, 1000, 1200}},
    {"extralarge", {1000, 2000, 2600}},
};

static void runPolybench(benchmark::State &state,
                         const std::vector<size_t> &size) {
  const size_t TMAX = size[0];
  const size_t NX = size[1];
  const size_t NY = size[2];

  MemRef<double, 2> inputEx({NX, NY}, 0);
  MemRef<double, 2> inputEy({NX, NY}, 0);
  MemRef<double, 2> inputHz({NX, NY}, 0);
  MemRef<double, 1> inputFict({TMAX}, 0);

  for (auto _ : state) {
    state.PauseTiming();
    _mlir_ciface_fdtd_2d_init_array(TMAX, NX, NY, &inputEx, &inputEy, &inputHz,
                                    &inputFict);
    state.ResumeTiming();
    _mlir_ciface_fdtd_2d(TMAX, NX, NY, &inputEx, &inputEy, &inputHz,
                         &inputFict);
  }
}

static void printArray(int nx, int ny, double *ex, double *ey, double *hz) {
  printf("begin dump: ex");
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", ex[i * ny + j]);
    }
  }
  printf("\nend   dump: ex\n");

  printf("begin dump: ey");
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", ey[i * ny + j]);
    }
  }
  printf("\nend   dump: ey\n");

  printf("begin dump: hz");
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", hz[i * ny + j]);
    }
  }
  printf("\nend   dump: hz\n");
}

void registerMLIRPolybenchFdtd2D(const std::set<std::string> &disabledSizes) {
  for (const auto &sizePair : sizes) {
    if (disabledSizes.count(sizePair.first)) {
      continue;
    }
    std::string benchmarkName = "fdtd-2d-" + sizePair.first;
    benchmark::RegisterBenchmark(benchmarkName.c_str(),
                                 [sizePair](benchmark::State &state) {
                                   runPolybench(state, sizePair.second);
                                 })
        ->Unit(benchmark::kMillisecond);
  }
}

void generateResultMLIRPolybenchFdtd2D(size_t size_id) {
  const std::string benchmarkName = "fdtd-2d-" + sizes[size_id].first;
  const std::vector<size_t> &size = sizes[size_id].second;

  const size_t TMAX = size[0];
  const size_t NX = size[1];
  const size_t NY = size[2];

  MemRef<double, 2> inputEx({NX, NY}, 0);
  MemRef<double, 2> inputEy({NX, NY}, 0);
  MemRef<double, 2> inputHz({NX, NY}, 0);
  MemRef<double, 1> inputFict({TMAX}, 0);

  _mlir_ciface_fdtd_2d_init_array(TMAX, NX, NY, &inputEx, &inputEy, &inputHz,
                                  &inputFict);

  _mlir_ciface_fdtd_2d(TMAX, NX, NY, &inputEx, &inputEy, &inputHz, &inputFict);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Result for " << benchmarkName << ":\n";
  printArray(NX, NY, inputEx.getData(), inputEy.getData(), inputHz.getData());
  std::cout << "--------------------------------------------------------"
            << std::endl;
}
