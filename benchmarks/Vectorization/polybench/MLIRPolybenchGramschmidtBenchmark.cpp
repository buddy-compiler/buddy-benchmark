//===- MLIRPolybenchGramschmidtBenchmark.cpp ------------------------------===//
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
// This file implements the gramschmidt Polybench benchmark.
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
void _mlir_ciface_gramschmidt_init_array(int, int, MemRef<double, 2> *,
                                         MemRef<double, 2> *,
                                         MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, MemRef<double, 2> *, MemRef<double, 2> *,
                            MemRef<double, 2> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {20, 30}, {60, 80}, {200, 240}, {1000, 1200}, {2000, 2600},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto m = size[0];
  auto n = size[1];

  MemRef<double, 2> A({m, n}, 0);
  MemRef<double, 2> R({n, n}, 0);
  MemRef<double, 2> Q({m, n}, 0);

  _mlir_ciface_gramschmidt_init_array(m, n, &A, &R, &Q);

  return std::make_tuple(m, n, std::move(A), std::move(R), std::move(Q));
}

// Runs the provided kernel for the gramschmidt benchmark.
static void MLIRPolybenchGramschmidt(benchmark::State &state,
                                     KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [m, n, A, R, Q] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(m, n, &A, &R, &Q);
  }
}

// Run the kernel and return the memrefs for verification.
static auto runMLIRPolybenchGramschmidt(KernelFunc kernel, size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [m, n, A, R, Q] = initializeMemRefs(size);
  kernel(m, n, &A, &R, &Q);
  return std::make_pair(std::move(R), std::move(Q));
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int m, int n, double *R, double *Q) {
  polybench::startDump();
  polybench::beginDump("R");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", R[i * n + j]);
    }
  }
  polybench::endDump("R");

  polybench::beginDump("Q");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", Q[i * n + j]);
    }
  }
  polybench::endDump("Q");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_gramschmidt_kernel_scalar(int, int, MemRef<double, 2> *,
                                            MemRef<double, 2> *,
                                            MemRef<double, 2> *);

void _mlir_ciface_gramschmidt_kernel_autovec(int, int, MemRef<double, 2> *,
                                             MemRef<double, 2> *,
                                             MemRef<double, 2> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchGramschmidt, scalar,
                  _mlir_ciface_gramschmidt_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchGramschmidt, autovec,
                  _mlir_ciface_gramschmidt_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchGramschmidt(size_t size_id) {
  const std::string benchmarkName =
      "gramschmidt-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto [refR, refQ] = runMLIRPolybenchGramschmidt(
      _mlir_ciface_gramschmidt_kernel_scalar, size_id);

  auto [vecR, vecQ] = runMLIRPolybenchGramschmidt(
      _mlir_ciface_gramschmidt_kernel_autovec, size_id);

  polybench::verify(refR.getData(), vecR.getData(), refR.getSize(),
                    "autovec " + benchmarkName + " (R)");
  polybench::verify(refQ.getData(), vecQ.getData(), refQ.getSize(),
                    "autovec " + benchmarkName + " (Q)");
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchGramschmidt(size_t size_id) {
  const std::string benchmarkName =
      "gramschmidt-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto [R, Q] = runMLIRPolybenchGramschmidt(
      _mlir_ciface_gramschmidt_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(Q.getSizes()[0], Q.getSizes()[1], R.getData(), Q.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
