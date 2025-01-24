//===- MLIRPolybenchDoitgenBenchmark.cpp ----------------------------------===//
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
// This file implements the doitgen Polybench benchmark.
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
void _mlir_ciface_doitgen_init_array(int, int, int, MemRef<double, 3> *,
                                     MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, int, MemRef<double, 3> *,
                            MemRef<double, 2> *, MemRef<double, 1> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {8, 10, 12}, {20, 25, 30}, {40, 50, 60}, {140, 150, 160}, {220, 250, 270},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto nq = size[0];
  auto nr = size[1];
  auto np = size[2];

  MemRef<double, 3> A({nr, nq, np}, 0);
  MemRef<double, 2> C4({np, np}, 0);
  MemRef<double, 1> sum({np}, 0);

  _mlir_ciface_doitgen_init_array(nr, nq, np, &A, &C4);

  return std::make_tuple(nr, nq, np, std::move(A), std::move(C4),
                         std::move(sum));
}

// Runs the provided kernel for the doitgen benchmark.
static void MLIRPolybenchDoitgen(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [nr, nq, np, A, C4, sum] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(nr, nq, np, &A, &C4, &sum);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 3> runMLIRPolybenchDoitgen(KernelFunc kernel,
                                                 size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [nr, nq, np, A, C4, sum] = initializeMemRefs(size);
  kernel(nr, nq, np, &A, &C4, &sum);
  return A;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int nr, int nq, int np, double *A) {
  polybench::startDump();
  polybench::beginDump("A");
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < nq; j++) {
      for (int k = 0; k < np; k++) {
        if ((i * nq * np + j * np + k) % 20 == 0) {
          printf("\n");
        }
        printf("%0.2lf ", A[i * nq * np + j * np + k]);
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
void _mlir_ciface_doitgen_kernel_scalar(int, int, int, MemRef<double, 3> *,
                                        MemRef<double, 2> *,
                                        MemRef<double, 1> *);

void _mlir_ciface_doitgen_kernel_autovec(int, int, int, MemRef<double, 3> *,
                                         MemRef<double, 2> *,
                                         MemRef<double, 1> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchDoitgen, scalar,
                  _mlir_ciface_doitgen_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchDoitgen, autovec,
                  _mlir_ciface_doitgen_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchDoitgen(size_t size_id) {
  const std::string benchmarkName =
      "doitgen-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refA =
      runMLIRPolybenchDoitgen(_mlir_ciface_doitgen_kernel_scalar, size_id);

  auto vecA =
      runMLIRPolybenchDoitgen(_mlir_ciface_doitgen_kernel_autovec, size_id);
  polybench::verify(refA.getData(), vecA.getData(), refA.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchDoitgen(size_t size_id) {
  const std::string benchmarkName =
      "doitgen-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto A = runMLIRPolybenchDoitgen(_mlir_ciface_doitgen_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(A.getSizes()[0], A.getSizes()[1], A.getSizes()[2], A.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
