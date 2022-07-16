//===- MLIROptBenchmark.cpp -----------------------------------------------===//
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
// This file implements the benchmark for GEMM operation.
//
//===----------------------------------------------------------------------===//

#include "Utils/Container.h"
#include <benchmark/benchmark.h>

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_gemm(MemRef<double, 2> *A, MemRef<double, 2> *B,
                       MemRef<double, 2> *C);
}

void BM_GEMM(benchmark::State &state) {
  long M = 2088, N = 2048, K = 2048;
  intptr_t sizesA[2] = {M, K};
  intptr_t sizesB[2] = {K, N};
  intptr_t sizesC[2] = {M, N};

  MemRef<double, 2> A(sizesA, 1.0);
  MemRef<double, 2> B(sizesB, 1.0);
  MemRef<double, 2> C(sizesC, 0.0);

  for (auto _ : state) {
    _mlir_ciface_gemm(&A, &B, &C);
  }
}

} // namespace

// Register benchmarking function with different arguments.
// BENCHMARK(BM_GEMM)->DenseRange(512, 56320, 512);
BENCHMARK(BM_GEMM);
