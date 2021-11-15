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
// This file implements the benchmark for depthwise conv2d(nhw) operation.
//
//===----------------------------------------------------------------------===//

#include "Utils/Container.h"
#include <benchmark/benchmark.h>

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_depthwise_conv2d_nhw(MemRef<float, 4> *input,
                                       MemRef<float, 3> *filter,
                                       MemRef<float, 4> *output);
}

intptr_t sizesInput[4] = {1, 3, 3, 2};
intptr_t stridesInput[4] = {18, 6, 2, 1};

intptr_t sizesFilter[3] = {2, 2, 2};
intptr_t stridesFilter[3] = {4, 2, 1};

intptr_t sizesOutput[4] = {1, 2, 2, 2};
intptr_t stridesOutput[4] = {8, 4, 2, 1};
// Create input, filter, and output.
MemRef<float, 4> inputMemRef(2.0, sizesInput, stridesInput);
MemRef<float, 3> filterMemRef(1.0, sizesFilter, stridesFilter);
MemRef<float, 4> outputMemRef(0.0, sizesOutput, stridesOutput);

// Define benchmark function.
void BM_DepthwiseConv2DNhw(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_depthwise_conv2d_nhw(&inputMemRef, &filterMemRef,
                                        &outputMemRef);
    }
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_DepthwiseConv2DNhw)->Arg(1);
BENCHMARK(BM_DepthwiseConv2DNhw)->Arg(4);

// Print result function.
void printResult() {
  // Clear the output memref.
  MemRef<float, 4> outputMemRef(0.0, sizesOutput, stridesOutput);
  // Run the mlir function.
  _mlir_ciface_depthwise_conv2d_nhw(&inputMemRef, &filterMemRef, &outputMemRef);
  // Print the output.
  std::cout << "Output: [ ";
  for (int i = 0; i < 8; ++i)
    std::cout << outputMemRef.aligned[i] << " ";
  std::cout << "]" << std::endl;
}
