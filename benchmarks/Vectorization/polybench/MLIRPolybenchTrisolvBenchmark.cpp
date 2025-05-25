//===- MLIRPolybenchTrisolvBenchmark.cpp ----------------------------------===//
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
// This file implements the trisolv Polybench benchmark.
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
void _mlir_ciface_trisolv_init_array(int, MemRef<double, 2> *,
                                     MemRef<double, 1> *, MemRef<double, 1> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, MemRef<double, 2> *, MemRef<double, 1> *,
                            MemRef<double, 1> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {40}, {120}, {400}, {2000}, {4000},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto n = size[0];

  MemRef<double, 2> L({n, n}, 0);
  MemRef<double, 1> x({n}, 0);
  MemRef<double, 1> b({n}, 0);

  _mlir_ciface_trisolv_init_array(n, &L, &x, &b);

  return std::make_tuple(n, std::move(L), std::move(x), std::move(b));
}

// Runs the provided kernel for the trisolv benchmark.
static void MLIRPolybenchTrisolv(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [n, L, x, b] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(n, &L, &x, &b);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 1> runMLIRPolybenchTrisolv(KernelFunc kernel,
                                                 size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [n, L, x, b] = initializeMemRefs(size);
  kernel(n, &L, &x, &b);
  return x;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int n, double *x) {
  polybench::startDump();
  polybench::beginDump("x");
  for (int i = 0; i < n; i++) {
    printf("%0.2lf ", x[i]);
    if (i % 20 == 0) {
      printf("\n");
    }
  }
  polybench::endDump("x");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_trisolv_kernel_scalar(int, MemRef<double, 2> *,
                                        MemRef<double, 1> *,
                                        MemRef<double, 1> *);

void _mlir_ciface_trisolv_kernel_autovec(int, MemRef<double, 2> *,
                                         MemRef<double, 1> *,
                                         MemRef<double, 1> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchTrisolv, scalar,
                  _mlir_ciface_trisolv_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchTrisolv, autovec,
                  _mlir_ciface_trisolv_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchTrisolv(size_t size_id) {
  const std::string benchmarkName =
      "trisolv-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refX =
      runMLIRPolybenchTrisolv(_mlir_ciface_trisolv_kernel_scalar, size_id);

  auto vecX =
      runMLIRPolybenchTrisolv(_mlir_ciface_trisolv_kernel_autovec, size_id);
  polybench::verify(refX.getData(), vecX.getData(), refX.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchTrisolv(size_t size_id) {
  const std::string benchmarkName =
      "trisolv-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto x = runMLIRPolybenchTrisolv(_mlir_ciface_trisolv_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(x.getSize(), x.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
