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

#include <buddy/core/Container.h>
#include <benchmark/benchmark.h>
#include <cmath>
#include <iostream>
#include <cstdlib>

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_conv2d(MemRef<float, 4> *A, MemRef<float, 4> *B,
                       MemRef<float, 4> *C);
}

void BM_CONV(benchmark::State &state) {
  long M = state.range(0), N = state.range(0), K = state.range(0);
  intptr_t sizesA[2] = {M, K};
  intptr_t sizesB[2] = {K, N};
  intptr_t sizesC[2] = {M, N};

  MemRef<float, 2> A(sizesA, 1.0);
  MemRef<float, 2> B(sizesB, 1.0);
  MemRef<float, 2> C(sizesC, 0);

  for (auto _ : state) {
    _mlir_ciface_conv2d(&A, &B, &C);
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_CONV)->DenseRange(64, 2048, 64);
