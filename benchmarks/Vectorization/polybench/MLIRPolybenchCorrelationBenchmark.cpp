//===- MLIRPolybenchCorrelationBenchmark.cpp ------------------------------===//
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
// This file implements the correlation Polybench benchmark.
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
void _mlir_ciface_correlation_init_array(int, int, MemRef<double, 1> *,
                                         MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, double, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 1> *,
                            MemRef<double, 1> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {28, 32}, {80, 100}, {240, 260}, {1200, 1400}, {2600, 3000},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto m = size[0];
  auto n = size[1];

  MemRef<double, 1> floatN({1}, 0);
  MemRef<double, 2> data({n, m}, 0);
  MemRef<double, 2> corr({m, m}, 0);
  MemRef<double, 1> mean({m}, 0);
  MemRef<double, 1> stddev({m}, 0);

  _mlir_ciface_correlation_init_array(m, n, &floatN, &data);

  return std::make_tuple(m, n, std::move(floatN), std::move(data),
                         std::move(corr), std::move(mean), std::move(stddev));
}

// Runs the provided kernel for the correlation benchmark.
static void MLIRPolybenchCorrelation(benchmark::State &state,
                                     KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [m, n, floatN, data, corr, mean, stddev] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(m, n, floatN.getData()[0], &data, &corr, &mean, &stddev);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 2> runMLIRPolybenchCorrelation(KernelFunc kernel,
                                                     size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [m, n, floatN, data, corr, mean, stddev] = initializeMemRefs(size);
  kernel(m, n, floatN.getData()[0], &data, &corr, &mean, &stddev);
  return corr;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int m, double *corr) {
  polybench::startDump();
  polybench::beginDump("corr");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", corr[i * m + j]);
    }
  }
  polybench::endDump("corr");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_correlation_kernel_scalar(int, int, double,
                                            MemRef<double, 2> *,
                                            MemRef<double, 2> *,
                                            MemRef<double, 1> *,
                                            MemRef<double, 1> *);

void _mlir_ciface_correlation_kernel_autovec(int, int, double,
                                             MemRef<double, 2> *,
                                             MemRef<double, 2> *,
                                             MemRef<double, 1> *,
                                             MemRef<double, 1> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchCorrelation, scalar,
                  _mlir_ciface_correlation_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchCorrelation, autovec,
                  _mlir_ciface_correlation_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchCorrelation(size_t size_id) {
  const std::string benchmarkName =
      "correlation-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refCorr = runMLIRPolybenchCorrelation(
      _mlir_ciface_correlation_kernel_scalar, size_id);

  auto vecCorr = runMLIRPolybenchCorrelation(
      _mlir_ciface_correlation_kernel_autovec, size_id);
  polybench::verify(refCorr.getData(), vecCorr.getData(), refCorr.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchCorrelation(size_t size_id) {
  const std::string benchmarkName =
      "correlation-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto corr = runMLIRPolybenchCorrelation(
      _mlir_ciface_correlation_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(corr.getSizes()[0], corr.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
