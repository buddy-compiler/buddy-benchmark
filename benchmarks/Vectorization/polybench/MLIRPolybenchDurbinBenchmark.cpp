//===- MLIRPolybenchDurbinBenchmark.cpp -----------------------------------===//
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
// This file implements the durbin Polybench benchmark.
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
void _mlir_ciface_durbin_init_array(int, MemRef<double, 1> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, MemRef<double, 1> *, MemRef<double, 1> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {40}, {120}, {400}, {2000}, {4000},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto n = size[0];

  MemRef<double, 1> r({n}, 0);
  MemRef<double, 1> y({n}, 0);

  _mlir_ciface_durbin_init_array(n, &r);

  return std::make_tuple(n, std::move(r), std::move(y));
}

// Runs the provided kernel for the durbin benchmark.
static void MLIRPolybenchDurbin(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [n, r, y] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(n, &r, &y);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 1> runMLIRPolybenchDurbin(KernelFunc kernel,
                                                size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [n, r, y] = initializeMemRefs(size);
  kernel(n, &r, &y);
  return y;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int n, double *y) {
  polybench::startDump();
  polybench::beginDump("y");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", y[i]);
  }
  polybench::endDump("y");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_durbin_kernel_scalar(int, MemRef<double, 1> *,
                                       MemRef<double, 1> *);

void _mlir_ciface_durbin_kernel_autovec(int, MemRef<double, 1> *,
                                        MemRef<double, 1> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchDurbin, scalar,
                  _mlir_ciface_durbin_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchDurbin, autovec,
                  _mlir_ciface_durbin_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchDurbin(size_t size_id) {
  const std::string benchmarkName =
      "durbin-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refY =
      runMLIRPolybenchDurbin(_mlir_ciface_durbin_kernel_scalar, size_id);

  auto vecY =
      runMLIRPolybenchDurbin(_mlir_ciface_durbin_kernel_autovec, size_id);
  polybench::verify(refY.getData(), vecY.getData(), refY.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchDurbin(size_t size_id) {
  const std::string benchmarkName =
      "durbin-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto y = runMLIRPolybenchDurbin(_mlir_ciface_durbin_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(y.getSize(), y.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
