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

#include "Utils.hpp"
#include "benchmark/benchmark.h"
#include "buddy/Core/Container.h"

#include <tuple>
#include <vector>

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

extern "C" {
// Initialization kernel for the benchmark. Not counted in execution time.
void _mlir_ciface_fdtd_2d_init_array(int, int, int, MemRef<double, 2> *,
                                     MemRef<double, 2> *, MemRef<double, 2> *,
                                     MemRef<double, 1> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, int, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 2> *,
                            MemRef<double, 1> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {20, 20, 30},      {40, 60, 80},       {100, 200, 240},
    {500, 1000, 1200}, {1000, 2000, 2600},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto tmax = size[0];
  auto nx = size[1];
  auto ny = size[2];

  MemRef<double, 2> ex({nx, ny}, 0);
  MemRef<double, 2> ey({nx, ny}, 0);
  MemRef<double, 2> hz({nx, ny}, 0);
  MemRef<double, 1> fict({tmax}, 0);

  _mlir_ciface_fdtd_2d_init_array(tmax, nx, ny, &ex, &ey, &hz, &fict);

  return std::make_tuple(tmax, nx, ny, std::move(ex), std::move(ey),
                         std::move(hz), std::move(fict));
}

// Runs the provided kernel for the fdtd-2d benchmark.
static void MLIRPolybenchFdtd2D(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [tmax, nx, ny, ex, ey, hz, fict] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(tmax, nx, ny, &ex, &ey, &hz, &fict);
  }
}

// Run the kernel and return the memref instance for verification.
static auto runMLIRPolybenchFdtd2D(KernelFunc kernel, size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [tmax, nx, ny, ex, ey, hz, fict] = initializeMemRefs(size);
  kernel(tmax, nx, ny, &ex, &ey, &hz, &fict);
  return std::make_tuple(std::move(ex), std::move(ey), std::move(hz));
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int nx, int ny, double *ex, double *ey, double *hz) {
  polybench::startDump();
  polybench::beginDump("ex");
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", ex[i * ny + j]);
    }
  }
  polybench::endDump("ex");
  polybench::finishDump();

  polybench::beginDump("ey");
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", ey[i * ny + j]);
    }
  }
  polybench::endDump("ey");

  polybench::beginDump("hz");
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", hz[i * ny + j]);
    }
  }
  polybench::endDump("hz");
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_fdtd_2d_kernel_scalar(int, int, int, MemRef<double, 2> *,
                                        MemRef<double, 2> *,
                                        MemRef<double, 2> *,
                                        MemRef<double, 1> *);

void _mlir_ciface_fdtd_2d_kernel_autovec(int, int, int, MemRef<double, 2> *,
                                         MemRef<double, 2> *,
                                         MemRef<double, 2> *,
                                         MemRef<double, 1> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchFdtd2D, scalar,
                  _mlir_ciface_fdtd_2d_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchFdtd2D, autovec,
                  _mlir_ciface_fdtd_2d_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchFdtd2D(size_t size_id) {
  const std::string benchmarkName =
      "fdtd-2d-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto [refEx, refEy, refHz] =
      runMLIRPolybenchFdtd2D(_mlir_ciface_fdtd_2d_kernel_scalar, size_id);

  auto [vecEx, vecEy, vecHz] =
      runMLIRPolybenchFdtd2D(_mlir_ciface_fdtd_2d_kernel_autovec, size_id);
  polybench::verify(refEx.getData(), vecEx.getData(), refEx.getSize(),
                    "autovec " + benchmarkName + " (ex)");
  polybench::verify(refEy.getData(), vecEy.getData(), refEy.getSize(),
                    "autovec " + benchmarkName + " (ey)");
  polybench::verify(refHz.getData(), vecHz.getData(), refHz.getSize(),
                    "autovec " + benchmarkName + " (hz)");
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchFdtd2D(size_t size_id) {
  const std::string benchmarkName =
      "fdtd-2d-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto [ex, ey, hz] =
      runMLIRPolybenchFdtd2D(_mlir_ciface_fdtd_2d_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(ex.getSizes()[0], ex.getSizes()[1], ex.getData(), ey.getData(),
             hz.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
