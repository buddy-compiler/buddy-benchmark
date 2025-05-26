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
#include "benchmark/benchmark.h"
#include "buddy/Core/Container.h"

#include <tuple>
#include <vector>

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

extern "C" {
// Initialization kernel for the benchmark. Not counted in execution time.
void _mlir_ciface_bicg_init_array(int, int, MemRef<double, 2> *,
                                  MemRef<double, 1> *, MemRef<double, 1> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, MemRef<double, 2> *, MemRef<double, 1> *,
                            MemRef<double, 1> *, MemRef<double, 1> *,
                            MemRef<double, 1> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {38, 42}, {116, 124}, {390, 410}, {1900, 2100}, {1800, 2200},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto m = size[0];
  auto n = size[1];

  MemRef<double, 2> A({n, m}, 0);
  MemRef<double, 1> s({m}, 0);
  MemRef<double, 1> q({n}, 0);
  MemRef<double, 1> p({m}, 0);
  MemRef<double, 1> r({n}, 0);

  _mlir_ciface_bicg_init_array(m, n, &A, &r, &p);

  return std::make_tuple(m, n, std::move(A), std::move(s), std::move(q),
                         std::move(p), std::move(r));
}

// Runs the provided kernel for the bicg benchmark.
static void MLIRPolybenchBicg(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [m, n, A, s, q, p, r] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(m, n, &A, &s, &q, &p, &r);
  }
}

// Run the kernel and return the memref instances for verification.
static auto runMLIRPolybenchBicg(KernelFunc kernel, size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [m, n, A, s, q, p, r] = initializeMemRefs(size);
  kernel(m, n, &A, &s, &q, &p, &r);
  return std::make_pair(std::move(s), std::move(q));
}

// Mimic the output format of the original Polybench implementation.
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

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_bicg_kernel_scalar(int, int, MemRef<double, 2> *,
                                     MemRef<double, 1> *, MemRef<double, 1> *,
                                     MemRef<double, 1> *, MemRef<double, 1> *);

void _mlir_ciface_bicg_kernel_autovec(int, int, MemRef<double, 2> *,
                                      MemRef<double, 1> *, MemRef<double, 1> *,
                                      MemRef<double, 1> *, MemRef<double, 1> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchBicg, scalar, _mlir_ciface_bicg_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchBicg, autovec, _mlir_ciface_bicg_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchBicg(size_t size_id) {
  const std::string benchmarkName =
      "bicg-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto [refS, refQ] =
      runMLIRPolybenchBicg(_mlir_ciface_bicg_kernel_scalar, size_id);

  auto [vecS, vecQ] =
      runMLIRPolybenchBicg(_mlir_ciface_bicg_kernel_autovec, size_id);
  polybench::verify(refS.getData(), vecS.getData(), refS.getSize(),
                    "autovec " + benchmarkName + " (s)");
  polybench::verify(refQ.getData(), vecQ.getData(), refQ.getSize(),
                    "autovec " + benchmarkName + " (q)");
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchBicg(size_t size_id) {
  const std::string benchmarkName =
      "bicg-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto [s, q] = runMLIRPolybenchBicg(_mlir_ciface_bicg_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(s.getSize(), q.getSize(), s.getData(), q.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
