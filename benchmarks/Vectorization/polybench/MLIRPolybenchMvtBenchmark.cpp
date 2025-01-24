//===- MLIRPolybenchMvtBenchmark.cpp --------------------------------------===//
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
// This file implements the mvt Polybench benchmark.
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
void _mlir_ciface_mvt_init_array(int, MemRef<double, 1> *, MemRef<double, 1> *,
                                 MemRef<double, 1> *, MemRef<double, 1> *,
                                 MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, MemRef<double, 1> *, MemRef<double, 1> *,
                            MemRef<double, 1> *, MemRef<double, 1> *,
                            MemRef<double, 2> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {40}, {120}, {400}, {2000}, {4000},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto n = size[0];

  MemRef<double, 1> x1({n}, 0);
  MemRef<double, 1> x2({n}, 0);
  MemRef<double, 1> y1({n}, 0);
  MemRef<double, 1> y2({n}, 0);
  MemRef<double, 2> A({n, n}, 0);

  _mlir_ciface_mvt_init_array(n, &x1, &x2, &y1, &y2, &A);

  return std::make_tuple(n, std::move(x1), std::move(x2), std::move(y1),
                         std::move(y2), std::move(A));
}

// Runs the provided kernel for the mvt benchmark.
static void MLIRPolybenchMvt(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [n, x1, x2, y1, y2, A] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(n, &x1, &x2, &y1, &y2, &A);
  }
}

// Run the kernel and return the memref instances for verification.
static auto runMLIRPolybenchMvt(KernelFunc kernel, size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [n, x1, x2, y1, y2, A] = initializeMemRefs(size);
  kernel(n, &x1, &x2, &y1, &y2, &A);
  return std::make_pair(std::move(x1), std::move(x2));
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int n, double *x1, double *x2) {
  polybench::startDump();
  polybench::beginDump("x1");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", x1[i]);
  }
  polybench::endDump("x1");

  polybench::beginDump("x2");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0) {
      printf("\n");
    }
    printf("%0.2lf ", x2[i]);
  }
  polybench::endDump("x2");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_mvt_kernel_scalar(int, MemRef<double, 1> *,
                                    MemRef<double, 1> *, MemRef<double, 1> *,
                                    MemRef<double, 1> *, MemRef<double, 2> *);

void _mlir_ciface_mvt_kernel_autovec(int, MemRef<double, 1> *,
                                     MemRef<double, 1> *, MemRef<double, 1> *,
                                     MemRef<double, 1> *, MemRef<double, 2> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchMvt, scalar, _mlir_ciface_mvt_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchMvt, autovec, _mlir_ciface_mvt_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchMvt(size_t size_id) {
  const std::string benchmarkName =
      "mvt-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto [refX1, refX2] =
      runMLIRPolybenchMvt(_mlir_ciface_mvt_kernel_scalar, size_id);

  auto [vecX1, vecX2] =
      runMLIRPolybenchMvt(_mlir_ciface_mvt_kernel_autovec, size_id);

  polybench::verify(refX1.getData(), vecX1.getData(), refX1.getSize(),
                    "autovec " + benchmarkName + " (x1)");
  polybench::verify(refX2.getData(), vecX2.getData(), refX2.getSize(),
                    "autovec " + benchmarkName + " (x2)");
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchMvt(size_t size_id) {
  const std::string benchmarkName =
      "mvt-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto [x1, x2] = runMLIRPolybenchMvt(_mlir_ciface_mvt_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(x1.getSize(), x1.getData(), x2.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
