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
    void _mlir_ciface_gemm(MemRef<float, 2> *A, MemRef<float, 2> *B, MemRef<float, 2> *C);
}

int M = 128;
int N = 128;
int K = 128;

intptr_t sizesA[2] = {M, K};
intptr_t sizesB[2] = {K, N};
intptr_t sizesC[2] = {M, N};

// Create input, filter, and output.
MemRef<float, 2> A(sizesA, 1.0);
MemRef<float, 2> B(sizesB, 1.0);
MemRef<float, 2> C(sizesC, 0.0);

// Define benchmark function.
void BM_GEMM(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_gemm(&A, &B, &C);
    }
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_GEMM)->Arg(1);
BENCHMARK(BM_GEMM)->Arg(4);

// Print result function.
void printResult() {
  // Clear the output memref.
  MemRef<float, 2> outputMemRef(sizesC, 0.0);
  // Run the mlir function.
  _mlir_ciface_gemm(&A, &B, &outputMemRef);
  // Print the output.
  std::cout << "Output: [ ";
  for (int i = 0; i < 8; ++i)
    std::cout << outputMemRef[i] << " ";
  std::cout << "]" << std::endl;
}
