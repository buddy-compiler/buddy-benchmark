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
// This file implements the benchmark for pointwise conv2d(nhwc-hwcf) operation.
//
//===----------------------------------------------------------------------===//

#include "Utils/Container.h"
#include <benchmark/benchmark.h>

// kNanosecond, kMicrosecond, kMillisecond, kSecond.
#define UNIT benchmark::kMillisecond
#define ITERATION 100

namespace {

// Declare the mobilenet C interface.
extern "C" void
_mlir_ciface_pointwise_conv_2d_nhwc_hwcf(const MemRef<float, 4> *input,
                                         const MemRef<float, 4> *filter,
                                         MemRef<float, 4> *output);
extern "C" MemRef<float, 4>
_mlir_ciface_pointwise_conv_2d_nhwc_hwcf_with_return(
    const MemRef<float, 4> *input, const MemRef<float, 4> *filter);
extern "C" MemRef<float, 4>
_mlir_ciface_pointwise_conv_2d_nhwc_hwcf_with_return_origin(
    const MemRef<float, 4> *input, const MemRef<float, 4> *filter);

intptr_t sizesInput[4] = {1, 4, 5, 2};
intptr_t sizesFilter[4] = {1, 1, 2, 7};
intptr_t sizesOutput[4] = {1, 4, 5, 7};

// Create input, filter, and output.
MemRef<float, 4> inputMemRef(sizesInput, 2);
MemRef<float, 4> filterMemRef(sizesFilter, 3);

MemRef<float, 4> inputMemReturn(sizesInput, 2);
MemRef<float, 4> filterMemReturn(sizesFilter, 3);

MemRef<float, 4> outputMemRef(sizesOutput, 0);
// Define benchmark function.void
void BM_PointwiseConv2DNhwcHwcf(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // MemRef<float, 4> outputMemRef(sizesOutput, 0);
      _mlir_ciface_pointwise_conv_2d_nhwc_hwcf(&inputMemRef, &filterMemRef,
                                               &outputMemRef);
    }
  }
}

MemRef<float, 4> outputMemReturn(sizesOutput, 0);
void BM_PointwiseConv2DNhwcHwcfReturn(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // MemRef<float, 4> outputMemReturn(sizesOutput, 0);
      outputMemReturn = _mlir_ciface_pointwise_conv_2d_nhwc_hwcf_with_return(
          &inputMemReturn, &filterMemReturn);
    }
  }
}

MemRef<float, 4> outputMemReturnOrigin(sizesOutput, 0);
void BM_PointwiseConv2DNhwcHwcfReturnOrigin(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      // MemRef<float, 4> outputMemReturn(sizesOutput, 0);
      outputMemReturnOrigin =
          _mlir_ciface_pointwise_conv_2d_nhwc_hwcf_with_return_origin(
              &inputMemReturn, &filterMemReturn);
    }
  }
}
// Register benchmarking function with different arguments.
BENCHMARK(BM_PointwiseConv2DNhwcHwcf)->Arg(ITERATION)->Unit(UNIT);
BENCHMARK(BM_PointwiseConv2DNhwcHwcfReturn)->Arg(ITERATION)->Unit(UNIT);
BENCHMARK(BM_PointwiseConv2DNhwcHwcfReturnOrigin)->Arg(ITERATION)->Unit(UNIT);
} // namespace

// Print result function.
void printResult() {
  // Clear the output memref.
  MemRef<float, 4> outputMemRef(sizesOutput, 0);
  // Run the mlir function.
  _mlir_ciface_pointwise_conv_2d_nhwc_hwcf(&inputMemRef, &filterMemRef,
                                           &outputMemRef);

  std::cout << "inputMemRef: " << inputMemRef << std::endl;
  std::cout << "filterMemRef: " << filterMemRef << std::endl;
  std::cout << "outputMemRef: " << outputMemRef << std::endl;

  MemRef<float, 4> outputMemReturn(sizesOutput, 0);
  // Run the mlir function.
  outputMemReturn = _mlir_ciface_pointwise_conv_2d_nhwc_hwcf_with_return(
      &inputMemReturn, &filterMemReturn);

  std::cout << "inputMemReturn: " << inputMemReturn << std::endl;
  std::cout << "filterMemReturn: " << filterMemReturn << std::endl;
  std::cout << "outputMemReturn: " << outputMemReturn << std::endl;

  MemRef<float, 4> outputMemReturnOrigin(sizesOutput, 0);
  // Run the mlir function.
  outputMemReturnOrigin = _mlir_ciface_pointwise_conv_2d_nhwc_hwcf_with_return(
      &inputMemReturn, &filterMemReturn);
  std::cout << "inputMemReturn: " << inputMemReturn << std::endl;
  std::cout << "filterMemReturn: " << filterMemReturn << std::endl;
  std::cout << "outputMemReturnOrigin: " << outputMemReturnOrigin << std::endl;
}
