//===- MLIRPolybenchDericheBenchmark.cpp ----------------------------------===//
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
// This file implements the deriche Polybench benchmark.
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
void _mlir_ciface_deriche_init_array(int, int, MemRef<float, 1> *,
                                     MemRef<float, 2> *, MemRef<float, 2> *);
}

// Kernel function signature for the benchmark.
using KernelFunc = void (*)(int, int, float, MemRef<float, 2> *,
                            MemRef<float, 2> *, MemRef<float, 2> *,
                            MemRef<float, 2> *);

// Dataset sizes for the benchmark.
static const std::vector<std::vector<size_t>> DATASET_SIZES{
    {64, 64}, {192, 128}, {720, 480}, {4096, 2160}, {7680, 4320},
};

// Initializes the memrefs for the benchmark.
static auto initializeMemRefs(const std::vector<size_t> &size) {
  auto w = size[0];
  auto h = size[1];

  MemRef<float, 1> alpha({1}, 0);
  MemRef<float, 2> imgIn({w, h}, 0);
  MemRef<float, 2> imgOut({w, h}, 0);
  MemRef<float, 2> y1({w, h}, 0);
  MemRef<float, 2> y2({w, h}, 0);

  _mlir_ciface_deriche_init_array(w, h, &alpha, &imgIn, &imgOut);

  return std::make_tuple(w, h, std::move(alpha), std::move(imgIn),
                         std::move(imgOut), std::move(y1), std::move(y2));
}

// Runs the provided kernel for the deriche benchmark.
static void MLIRPolybenchDeriche(benchmark::State &state, KernelFunc kernel) {
  // The dataset size is determined by the argument passed by google benchmark.
  const auto &size = DATASET_SIZES[state.range(0)];
  for (auto _ : state) {
    state.PauseTiming();
    // Skip the initialization time from the measurement.
    auto [w, h, alpha, imgIn, imgOut, y1, y2] = initializeMemRefs(size);
    state.ResumeTiming();
    kernel(w, h, alpha.getData()[0], &imgIn, &imgOut, &y1, &y2);
  }
}

// Run the kernel and return the memref instance for verification.
static MemRef<float, 2> runMLIRPolybenchDeriche(KernelFunc kernel,
                                                size_t size_id) {
  const auto &size = DATASET_SIZES[size_id];
  auto [w, h, alpha, imgIn, imgOut, y1, y2] = initializeMemRefs(size);
  kernel(w, h, alpha.getData()[0], &imgIn, &imgOut, &y1, &y2);
  return imgOut;
}

// Mimic the output format of the original Polybench implementation.
static void printArray(int w, int h, float *imgOut) {
  polybench::startDump();
  polybench::beginDump("imgOut");
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) {
        printf("\n");
      }
      printf("%0.2f ", imgOut[i * h + j]);
    }
  }
  polybench::endDump("imgOut");
  polybench::finishDump();
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. New methods can be added here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_deriche_kernel_scalar(int, int, float, MemRef<float, 2> *,
                                        MemRef<float, 2> *, MemRef<float, 2> *,
                                        MemRef<float, 2> *);

void _mlir_ciface_deriche_kernel_autovec(int, int, float, MemRef<float, 2> *,
                                         MemRef<float, 2> *, MemRef<float, 2> *,
                                         MemRef<float, 2> *);
/// [Step 1] Add function of new methods here.
}

BENCHMARK_CAPTURE(MLIRPolybenchDeriche, scalar,
                  _mlir_ciface_deriche_kernel_scalar)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(MLIRPolybenchDeriche, autovec,
                  _mlir_ciface_deriche_kernel_autovec)
    ->DenseRange(0, DATASET_SIZES.size() - 1)
    ->Unit(benchmark::kMillisecond);
/// [Step 2] Register new benchmarks here.

void verifyResultMLIRPolybenchDeriche(size_t size_id) {
  const std::string benchmarkName =
      "deriche-" + polybench::getPolybenchDatasetSizeName(size_id);

  auto refImgOut =
      runMLIRPolybenchDeriche(_mlir_ciface_deriche_kernel_scalar, size_id);

  auto vecImgOut =
      runMLIRPolybenchDeriche(_mlir_ciface_deriche_kernel_autovec, size_id);
  polybench::verify(refImgOut.getData(), vecImgOut.getData(),
                    refImgOut.getSize(), "autovec " + benchmarkName);
  // [Step 3] Add verification code here.
}

// -----------------------------------------------------------------------------
// Additional utility functions. No need to change the code here.
// -----------------------------------------------------------------------------

// Generate the baseline result for the benchmark to verify the correctness of
// the ported code.
void generateResultMLIRPolybenchDeriche(size_t size_id) {
  const std::string benchmarkName =
      "deriche-" + polybench::getPolybenchDatasetSizeName(size_id);
  auto imgOut =
      runMLIRPolybenchDeriche(_mlir_ciface_deriche_kernel_scalar, size_id);
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Result for " << benchmarkName << ":" << std::endl;
  printArray(imgOut.getSizes()[0], imgOut.getSizes()[1], imgOut.getData());
  std::cout << "------------------------------------------------" << std::endl;
}
