//===- MLIRPolybenchAdiBenchmark.cpp --------------------------------------===//
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
// This file implements the adi Polybench benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include "benchmark/benchmark.h"
#include "buddy/Core/Container.h"

#include <tuple>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_adi_init_array(int, MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, MemRef<double, 2> *, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 2> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {20, 20}, {40, 60}, {100, 200}, {500, 1000}, {1000, 2000},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto tsteps = size[0];
  auto n = size[1];

  MemRef<double, 2> u({n, n}, 0);
  MemRef<double, 2> v({n, n}, 0);
  MemRef<double, 2> p({n, n}, 0);
  MemRef<double, 2> q({n, n}, 0);

  _mlir_ciface_adi_init_array(n, &u);

  return std::make_tuple(tsteps, n, std::move(u), std::move(v), std::move(p),
                         std::move(q));
}

// Runs the provided kernel for the adi benchmark.
static void MLIRPolybenchAdi(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [tsteps, n, u, v, p, q] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(tsteps, n, &u, &v, &p, &q);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 2> runMLIRPolybenchAdi(KernelFunc kernel,
                                             size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [tsteps, n, u, v, p, q] = initializeMemRefs(size);
  kernel(tsteps, n, &u, &v, &p, &q);
  return u;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int n, double *u) {
  polybench::startDump();
  polybench::beginDump("u");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", u[i * n + j]);
    }
  }
  polybench::endDump("u");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_adi_kernel_scalar(int, int, MemRef<double, 2> *,
                                    MemRef<double, 2> *, MemRef<double, 2> *,
                                    MemRef<double, 2> *);
void _mlir_ciface_adi_kernel_autovec(int, int, MemRef<double, 2> *,
                                     MemRef<double, 2> *, MemRef<double, 2> *,
                                     MemRef<double, 2> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchAdi, scalar, _mlir_ciface_adi_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchAdi, autovec, _mlir_ciface_adi_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchAdi(size_t size_id) {
  const std::string benchmarkName =
      "adi-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refU = runMLIRPolybenchAdi(_mlir_ciface_adi_kernel_scalar, size_id);

  auto vecU = runMLIRPolybenchAdi(_mlir_ciface_adi_kernel_autovec, size_id);
  polybench::verify(refU.getData(), vecU.getData(), refU.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchAdi(size_t size_id) {
  const std::string benchmarkName =
      "adi-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto u = runMLIRPolybenchAdi(_mlir_ciface_adi_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(u.getSizes()[0], u.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
