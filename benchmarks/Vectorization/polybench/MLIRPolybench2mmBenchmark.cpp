//===- MLIRPolybench2mmBenchmark.cpp --------------------------------------===//
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
// This file implements the 2mm Polybench benchmark.
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
// Initialization kernel for the benchmark. This function is not counted in the
// benchmark execution time, so it is not necessary to optimize it.
void _mlir_ciface_polybench_2mm_init_array(
    int, int, int, int, MemRef<double, 1> *, MemRef<double, 1> *,
    MemRef<double, 2> *, MemRef<double, 2> *, MemRef<double, 2> *,
    MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, int, int, double, double,
                            MemRef<double, 2> *, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 2> *,
                            MemRef<double, 2> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {16, 18, 22, 24},       {40, 50, 70, 80},         {180, 190, 210, 220},
    {800, 900, 1100, 1200}, {1600, 1800, 2200, 2400},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto ni = size[0];
  auto nj = size[1];
  auto nk = size[2];
  auto nl = size[3];

  MemRef<double, 1> alpha({1}, 0);
  MemRef<double, 1> beta({1}, 0);
  MemRef<double, 2> tmp({ni, nj}, 0);
  MemRef<double, 2> A({ni, nk}, 0);
  MemRef<double, 2> B({nk, nj}, 0);
  MemRef<double, 2> C({nj, nl}, 0);
  MemRef<double, 2> D({ni, nl}, 0);

  _mlir_ciface_polybench_2mm_init_array(ni, nj, nk, nl, &alpha, &beta, &A, &B,
                                        &C, &D);

  return std::make_tuple(ni, nj, nk, nl, std::move(alpha), std::move(beta),
                         std::move(tmp), std::move(A), std::move(B),
                         std::move(C), std::move(D));
}

// Runs the provided kernel for the 2mm benchmark.
static void MLIRPolybench2mm(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D] =
        initializeMemRefs(size);
    state.ResumeTiming();
    kernel(ni, nj, nk, nl, alpha.getData()[0], beta.getData()[0], &tmp, &A, &B,
           &C, &D);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 2> runMLIRPolybench2mm(KernelFunc kernel,
                                             size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [NI, nj, nk, nl, alpha, beta, tmp, A, B, C, D] = initializeMemRefs(size);
  kernel(NI, nj, nk, nl, alpha.getData()[0], beta.getData()[0], &tmp, &A, &B,
         &C, &D);
  return D;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int ni, int nl, double *D) {
  polybench::startDump();
  polybench::beginDump("D");
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", D[i * nl + j]);
    }
  }
  polybench::endDump("D");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_polybench_2mm_kernel_scalar(int, int, int, int, double,
                                              double, MemRef<double, 2> *,
                                              MemRef<double, 2> *,
                                              MemRef<double, 2> *,
                                              MemRef<double, 2> *,
                                              MemRef<double, 2> *);

void _mlir_ciface_polybench_2mm_kernel_autovec(int, int, int, int, double,
                                               double, MemRef<double, 2> *,
                                               MemRef<double, 2> *,
                                               MemRef<double, 2> *,
                                               MemRef<double, 2> *,
                                               MemRef<double, 2> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybench2mm, scalar,
                  _mlir_ciface_polybench_2mm_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybench2mm, autovec,
                  _mlir_ciface_polybench_2mm_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

// Verify the result on the given dataset size.
void verifyResultMLIRPolybench2mm(size_t size_id) {
  const std::string benchmarkName =
      "2mm-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refD =
      runMLIRPolybench2mm(_mlir_ciface_polybench_2mm_kernel_scalar, size_id);

  auto vecD =
      runMLIRPolybench2mm(_mlir_ciface_polybench_2mm_kernel_autovec, size_id);
  polybench::verify(refD.getData(), vecD.getData(), refD.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybench2mm(size_t size_id) {
  const std::string benchmarkName =
      "2mm-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto D =
      runMLIRPolybench2mm(_mlir_ciface_polybench_2mm_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(D.getSizes()[0], D.getSizes()[1], D.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
