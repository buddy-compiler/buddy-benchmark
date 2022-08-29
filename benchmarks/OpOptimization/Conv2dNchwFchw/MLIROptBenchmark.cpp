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
void _mlir_ciface_conv2d(MemRef<float, 4> *input, MemRef<float, 4> *filter,
                       MemRef<float, 4> *output);
}

void BM_CONV(benchmark::State &state) {
  long factor = state.range(0);
  long a = 1, b = factor, c = 16 * factor, d = 16 * factor,
       e = 1, f = 32 * factor, g = 32 * factor;

  intptr_t sizesInput[4] = {a, e, c + f, d + g};
  intptr_t sizesFilter[4] = {a, e, f, g};
  intptr_t sizesOutput[4] = {a, b, c, d};

  MemRef<float, 4> input(sizesInput, 1.0);
  MemRef<float, 4> filter(sizesFilter, 1.0);
  MemRef<float, 4> output(sizesOutput, 0);

  for (auto _ : state) {
	  _mlir_ciface_conv2d(&input, &filter, &output);
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_CONV)->DenseRange(10, 100, 10);
