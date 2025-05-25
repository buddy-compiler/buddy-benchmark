//===- MLIRPolybench3mmBenchmark.cpp --------------------------------------===//
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
// This file implements the 3mm Polybench benchmark.
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
void _mlir_ciface_polybench_3mm_init_array(int, int, int, int, int,
                                           MemRef<double, 2> *,
                                           MemRef<double, 2> *,
                                           MemRef<double, 2> *,
                                           MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, int, int, int, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 2> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {16, 18, 20, 22, 24},           {40, 50, 60, 70, 80},
    {180, 190, 200, 210, 220},      {800, 900, 1000, 1100, 1200},
    {1600, 1800, 2000, 2200, 2400},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto ni = size[0];
  auto nj = size[1];
  auto nk = size[2];
  auto nl = size[3];
  auto nm = size[4];

  MemRef<double, 2> E({ni, nj}, 0);
  MemRef<double, 2> A({ni, nk}, 0);
  MemRef<double, 2> B({nk, nj}, 0);
  MemRef<double, 2> F({nj, nl}, 0);
  MemRef<double, 2> C({nj, nm}, 0);
  MemRef<double, 2> D({nm, nl}, 0);
  MemRef<double, 2> G({ni, nl}, 0);

  _mlir_ciface_polybench_3mm_init_array(ni, nj, nk, nl, nm, &A, &B, &C, &D);

  return std::make_tuple(ni, nj, nk, nl, nm, std::move(E), std::move(A),
                         std::move(B), std::move(F), std::move(C), std::move(D),
                         std::move(G));
}

// Runs the provided kernel for the 3mm benchmark.
static void MLIRPolybench3mm(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [ni, nj, nk, nl, nm, E, A, B, F, C, D, G] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(ni, nj, nk, nl, nm, &E, &A, &B, &F, &C, &D, &G);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 2> runMLIRPolybench3mm(KernelFunc kernel,
                                             size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [ni, nj, nk, nl, nm, E, A, B, F, C, D, G] = initializeMemRefs(size);
  kernel(ni, nj, nk, nl, nm, &E, &A, &B, &F, &C, &D, &G);
  return G;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int ni, int nl, double *G) {
  polybench::startDump();
  polybench::beginDump("G");
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", G[i * nl + j]);
    }
  }
  polybench::endDump("G");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_polybench_3mm_kernel_scalar(
    int, int, int, int, int, MemRef<double, 2> *, MemRef<double, 2> *,
    MemRef<double, 2> *, MemRef<double, 2> *, MemRef<double, 2> *,
    MemRef<double, 2> *, MemRef<double, 2> *);
void _mlir_ciface_polybench_3mm_kernel_autovec(
    int, int, int, int, int, MemRef<double, 2> *, MemRef<double, 2> *,
    MemRef<double, 2> *, MemRef<double, 2> *, MemRef<double, 2> *,
    MemRef<double, 2> *, MemRef<double, 2> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybench3mm, scalar,
                  _mlir_ciface_polybench_3mm_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybench3mm, autovec,
                  _mlir_ciface_polybench_3mm_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybench3mm(size_t size_id) {
  const std::string benchmarkName =
      "3mm-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refG =
      runMLIRPolybench3mm(_mlir_ciface_polybench_3mm_kernel_scalar, size_id);

  auto vecG =
      runMLIRPolybench3mm(_mlir_ciface_polybench_3mm_kernel_autovec, size_id);
  polybench::verify(refG.getData(), vecG.getData(), refG.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybench3mm(size_t size_id) {
  const std::string benchmarkName =
      "3mm-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto G =
      runMLIRPolybench3mm(_mlir_ciface_polybench_3mm_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(G.getSizes()[0], G.getSizes()[1], G.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
