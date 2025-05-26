//===- MLIRPolybenchSeidel2DBenchmark.cpp ---------------------------------===//
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
// This file implements the seidel-2d Polybench benchmark.
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
void _mlir_ciface_seidel_2d_init_array(int, MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, MemRef<double, 2> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {20, 40}, {40, 120}, {100, 400}, {500, 2000}, {1000, 4000},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto tsteps = size[0];
  auto n = size[1];

  MemRef<double, 2> A({n, n}, 0);
  _mlir_ciface_seidel_2d_init_array(n, &A);

  return std::make_tuple(tsteps, n, std::move(A));
}

// Runs the provided kernel for the seidel-2d benchmark.
static void MLIRPolybenchSeidel2D(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [tsteps, n, A] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(tsteps, n, &A);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 2> runMLIRPolybenchSeidel2D(KernelFunc kernel,
                                                  size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [tsteps, n, A] = initializeMemRefs(size);
  kernel(tsteps, n, &A);
  return A;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int n, double *A) {
  polybench::startDump();
  polybench::beginDump("A");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", A[i * n + j]);
    }
  }
  polybench::endDump("A");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_seidel_2d_kernel_scalar(int, int, MemRef<double, 2> *);
void _mlir_ciface_seidel_2d_kernel_autovec(int, int, MemRef<double, 2> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchSeidel2D, scalar,
                  _mlir_ciface_seidel_2d_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchSeidel2D, autovec,
                  _mlir_ciface_seidel_2d_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchSeidel2D(size_t size_id) {
  const std::string benchmarkName =
      "seidel-2d-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refA =
      runMLIRPolybenchSeidel2D(_mlir_ciface_seidel_2d_kernel_scalar, size_id);

  auto vecA =
      runMLIRPolybenchSeidel2D(_mlir_ciface_seidel_2d_kernel_autovec, size_id);
  polybench::verify(refA.getData(), vecA.getData(), refA.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchSeidel2D(size_t size_id) {
  const std::string benchmarkName =
      "seidel-2d-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto A =
      runMLIRPolybenchSeidel2D(_mlir_ciface_seidel_2d_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(A.getSizes()[0], A.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
