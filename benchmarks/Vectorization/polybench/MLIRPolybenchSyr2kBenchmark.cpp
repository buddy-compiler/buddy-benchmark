//===- MLIRPolybenchSyr2kBenchmark.cpp ------------------------------------===//
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
// This file implements the syr2k Polybench benchmark.
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
void _mlir_ciface_syr2k_init_array(int, int, MemRef<double, 1> *,
                                   MemRef<double, 1> *, MemRef<double, 2> *,
                                   MemRef<double, 2> *, MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, double, double, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 2> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {20, 30}, {60, 80}, {200, 240}, {1000, 1200}, {2000, 2600},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto m = size[0];
  auto n = size[1];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);
  MemRef<double, 2> C({n, n}, 0);
  MemRef<double, 2> A({n, m}, 0);
  MemRef<double, 2> B({n, m}, 0);

  _mlir_ciface_syr2k_init_array(n, m, &alpha, &beta, &C, &A, &B);

  return std::make_tuple(m, n, std::move(alpha), std::move(beta), std::move(C),
                         std::move(A), std::move(B));
}

// Runs the provided kernel for the syr2k benchmark.
static void MLIRPolybenchSyr2k(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [m, n, alpha, beta, C, A, B] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(n, m, alpha.getData()[0], beta.getData()[0], &C, &A, &B);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 2> runMLIRPolybenchSyr2k(KernelFunc kernel,
                                               size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [m, n, alpha, beta, C, A, B] = initializeMemRefs(size);
  kernel(n, m, alpha.getData()[0], beta.getData()[0], &C, &A, &B);
  return C;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int n, double *C) {
  int i, j;
  polybench::startDump();
  polybench::beginDump("C");
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", C[i * n + j]);
    }
  }
  polybench::endDump("C");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_syr2k_kernel_scalar(int, int, double, double,
                                      MemRef<double, 2> *, MemRef<double, 2> *,
                                      MemRef<double, 2> *);

void _mlir_ciface_syr2k_kernel_autovec(int, int, double, double,
                                       MemRef<double, 2> *, MemRef<double, 2> *,
                                       MemRef<double, 2> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchSyr2k, scalar, _mlir_ciface_syr2k_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchSyr2k, autovec,
                  _mlir_ciface_syr2k_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchSyr2k(size_t size_id) {
  const std::string benchmarkName =
      "syr2k-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refC = runMLIRPolybenchSyr2k(_mlir_ciface_syr2k_kernel_scalar, size_id);

  auto vecC = runMLIRPolybenchSyr2k(_mlir_ciface_syr2k_kernel_autovec, size_id);
  polybench::verify(refC.getData(), vecC.getData(), refC.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchSyr2k(size_t size_id) {
  const std::string benchmarkName =
      "syr2k-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto C = runMLIRPolybenchSyr2k(_mlir_ciface_syr2k_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(C.getSizes()[0], C.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
