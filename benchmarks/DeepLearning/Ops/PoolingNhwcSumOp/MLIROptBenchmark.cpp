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
// This file implements the benchmark for sum pooling (nhwc) operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <buddy/DIP/ImageContainer.h>

namespace {

// Declare the pooling_nhwc_sum interface.
extern "C" {
void _mlir_ciface_pooling_nhwc_sum(MemRef<float, 4> *input,
                                   MemRef<float, 2> *filter,
                                   MemRef<float, 4> *output);
}

// Create input, filter, and output.
MemRef<float, 4> input({1, 6, 6, 1}, 1.0);
MemRef<float, 2> filter({3, 3}, 1.0);
intptr_t sizesOutput[4] = {1, 3, 3, 1};
MemRef<float, 4> output(sizesOutput);

// Define benchmark function.
void BM_PoolingNhwcSum(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_pooling_nhwc_sum(&input, &filter, &output);
    }
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_PoolingNhwcSum)->Arg(1);
BENCHMARK(BM_PoolingNhwcSum)->Arg(4);

// Print result function.
void printResult() {
  // Create the output memref.
  MemRef<float, 4> output(sizesOutput);
  // Run the mlir function.
  _mlir_ciface_pooling_nhwc_sum(&input, &filter, &output);
  // Print the output.
  std::cout << "Output: [ ";
  for (int i = 0; i < 9; ++i)
    std::cout << output[i] << " ";
  std::cout << "]" << std::endl;
}
