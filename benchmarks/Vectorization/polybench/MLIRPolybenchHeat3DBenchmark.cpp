//===- MLIRPolybenchHeat3DBenchmark.cpp -----------------------------------===//
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
// This file implements the heat-3d Polybench benchmark.
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
void _mlir_ciface_heat_3d_init_array(int, MemRef<double, 3> *,
                                     MemRef<double, 3> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, MemRef<double, 3> *, MemRef<double, 3> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {20, 10}, {40, 20}, {100, 40}, {500, 120}, {1000, 200},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto tsteps = size[0];
  auto n = size[1];

  MemRef<double, 3> A({n, n, n}, 0);
  MemRef<double, 3> B({n, n, n}, 0);

  _mlir_ciface_heat_3d_init_array(n, &A, &B);

  return std::make_tuple(tsteps, n, std::move(A), std::move(B));
}

// Runs the provided kernel for the heat-3d benchmark.
static void MLIRPolybenchHeat3D(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [tsteps, n, A, B] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(tsteps, n, &A, &B);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 3> runMLIRPolybenchHeat3D(KernelFunc kernel,
                                                size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [tsteps, n, A, B] = initializeMemRefs(size);
  kernel(tsteps, n, &A, &B);
  return A;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int n, double *A) {
  polybench::startDump();
  polybench::beginDump("A");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        if ((i * n * n + j * n + k) % 20 == 0) {
          printf("\n");
        }
        printf("%0.2lf ", A[i * n * n + j * n + k]);
      }
    }
  }
  polybench::endDump("A");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_heat_3d_kernel_scalar(int, int, MemRef<double, 3> *,
                                        MemRef<double, 3> *);

void _mlir_ciface_heat_3d_kernel_autovec(int, int, MemRef<double, 3> *,
                                         MemRef<double, 3> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchHeat3D, scalar,
                  _mlir_ciface_heat_3d_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchHeat3D, autovec,
                  _mlir_ciface_heat_3d_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchHeat3D(size_t size_id) {
  const std::string benchmarkName =
      "heat-3d-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refA =
      runMLIRPolybenchHeat3D(_mlir_ciface_heat_3d_kernel_scalar, size_id);

  auto vecA =
      runMLIRPolybenchHeat3D(_mlir_ciface_heat_3d_kernel_autovec, size_id);
  polybench::verify(refA.getData(), vecA.getData(), refA.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchHeat3D(size_t size_id) {
  const std::string benchmarkName =
      "heat-3d-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto A = runMLIRPolybenchHeat3D(_mlir_ciface_heat_3d_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(A.getSizes()[0], A.getData());
  std::cout << "------------------------------------------------" << std::endl;
}