//===- MLIRPolybenchAtaxBenchmark.cpp -------------------------------------===//
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
// This file implements the atax Polybench benchmark.
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
void _mlir_ciface_atax_init_array(int, int, MemRef<double, 2> *,
                                  MemRef<double, 1> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, MemRef<double, 2> *, MemRef<double, 1> *,
                            MemRef<double, 1> *, MemRef<double, 1> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {38, 42}, {116, 124}, {390, 410}, {1900, 2100}, {1800, 2200},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto m = size[0];
  auto n = size[1];

  MemRef<double, 2> A({m, n}, 0);
  MemRef<double, 1> x({n}, 0);
  MemRef<double, 1> y({n}, 0);
  MemRef<double, 1> tmp({m}, 0);

  _mlir_ciface_atax_init_array(m, n, &A, &x);

  return std::make_tuple(m, n, std::move(A), std::move(x), std::move(y),
                         std::move(tmp));
}

// Runs the provided kernel for the atax benchmark.
static void MLIRPolybenchAtax(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [m, n, A, x, y, tmp] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(m, n, &A, &x, &y, &tmp);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 1> runMLIRPolybenchAtax(KernelFunc kernel,
                                              size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [m, n, A, x, y, tmp] = initializeMemRefs(size);
  kernel(m, n, &A, &x, &y, &tmp);
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
void _mlir_ciface_atax_kernel_scalar(int, int, MemRef<double, 2> *,
                                     MemRef<double, 1> *, MemRef<double, 1> *,
                                     MemRef<double, 1> *);

void _mlir_ciface_atax_kernel_autovec(int, int, MemRef<double, 2> *,
                                      MemRef<double, 1> *, MemRef<double, 1> *,
                                      MemRef<double, 1> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchAtax, scalar, _mlir_ciface_atax_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchAtax, autovec, _mlir_ciface_atax_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchAtax(size_t size_id) {
  const std::string benchmarkName =
      "atax-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refY = runMLIRPolybenchAtax(_mlir_ciface_atax_kernel_scalar, size_id);

  auto vecY = runMLIRPolybenchAtax(_mlir_ciface_atax_kernel_autovec, size_id);
  polybench::verify(refY.getData(), vecY.getData(), refY.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchAtax(size_t size_id) {
  const std::string benchmarkName =
      "atax-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto y = runMLIRPolybenchAtax(_mlir_ciface_atax_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(y.getSizes()[0], y.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
