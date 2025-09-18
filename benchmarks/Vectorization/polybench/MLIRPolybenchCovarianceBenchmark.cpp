//===- MLIRPolybenchCovarianceBenchmark.cpp -------------------------------===//
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
// This file implements the covariance Polybench benchmark.
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
void _mlir_ciface_covariance_init_array(int, int, MemRef<double, 1> *,
                                        MemRef<double, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, double, MemRef<double, 2> *,
                            MemRef<double, 2> *, MemRef<double, 1> *);

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
  MemRef<double, 2> cov({m, m}, 0);
  MemRef<double, 1> mean({m}, 0);

  _mlir_ciface_covariance_init_array(m, n, &floatN, &data);

  return std::make_tuple(m, n, std::move(floatN), std::move(data),
                         std::move(cov), std::move(mean));
}

// Runs the provided kernel for the covariance benchmark.
static void MLIRPolybenchCovariance(benchmark::State &state,
                                    KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [m, n, floatN, data, cov, mean] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(m, n, floatN.getData()[0], &data, &cov, &mean);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<double, 2> runMLIRPolybenchCovariance(KernelFunc kernel,
                                                    size_t arg) {
  const auto &size = DATASET_SIZES[arg];
  auto [m, n, floatN, data, cov, mean] = initializeMemRefs(size);
  kernel(m, n, floatN.getData()[0], &data, &cov, &mean);
  return cov;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int m, double *cov) {
  polybench::startDump();
  polybench::beginDump("cov");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2lf ", cov[i * m + j]);
    }
  }
  polybench::endDump("cov");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_covariance_kernel_scalar(int, int, double,
                                           MemRef<double, 2> *,
                                           MemRef<double, 2> *,
                                           MemRef<double, 1> *);

void _mlir_ciface_covariance_kernel_autovec(int, int, double,
                                            MemRef<double, 2> *,
                                            MemRef<double, 2> *,
                                            MemRef<double, 1> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchCovariance, scalar,
                  _mlir_ciface_covariance_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchCovariance, autovec,
                  _mlir_ciface_covariance_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchCovariance(size_t size_id) {
  const std::string benchmarkName =
      "covariance-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refCov = runMLIRPolybenchCovariance(
      _mlir_ciface_covariance_kernel_scalar, size_id);

  auto vecCov = runMLIRPolybenchCovariance(
      _mlir_ciface_covariance_kernel_autovec, size_id);
  polybench::verify(refCov.getData(), vecCov.getData(), refCov.getSize(),
                    "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchCovariance(size_t arg) {
  const std::string benchmarkName =
      "covariance-" + polybench::getPolybenchDatasetSizeName(arg);
  auto cov =
      runMLIRPolybenchCovariance(_mlir_ciface_covariance_kernel_scalar, arg);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(cov.getSizes()[0], cov.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
