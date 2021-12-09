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
// This file implements the benchmark for conv2d(nhwc-hwcf) operation.
//
//===----------------------------------------------------------------------===//

#include "Utils/Container.h"
#include <benchmark/benchmark.h>

namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_conv_2d_nhwc_hwcf(MemRef<float, 4> *input,
                                    MemRef<float, 4> *filter,
                                    MemRef<float, 4> *output);
}

intptr_t sizesInput[4] = {1, 3, 3, 2};
intptr_t sizesFilter[4] = {2, 2, 2, 2};
intptr_t sizesOutput[4] = {1, 2, 2, 2};

// Create input, filter, and output.
MemRef<float, 4> inputMemRef(sizesInput, 2.0);
MemRef<float, 4> filterMemRef(sizesFilter, 3.0);
MemRef<float, 4> outputMemRef(sizesOutput, 0.0);

// Define benchmark function.
void BM_Conv2DNhwcHwcf(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_conv_2d_nhwc_hwcf(&inputMemRef, &filterMemRef,
                                     &outputMemRef);
    }
  }
}

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_Conv2DNhwcHwcf)->Arg(1);
BENCHMARK(BM_Conv2DNhwcHwcf)->Arg(4);

// Print result function.
void printResult() {
  // Clear the output memref.
  MemRef<float, 4> outputMemRef(sizesOutput, 0.0);
  // Run the mlir function.
  _mlir_ciface_conv_2d_nhwc_hwcf(&inputMemRef, &filterMemRef, &outputMemRef);
  // Print the output.
  std::cout << "Output: [ ";
  for (int i = 0; i < 8; ++i)
    std::cout << outputMemRef[i] << " ";
  std::cout << "]" << std::endl;
}
