//===- Main.cpp -----------------------------------------------------------===//
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
// This is the main file of the Depthwise Conv2D NHWC-HWC benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the number here as needed.
// -----------------------------------------------------------------------------

#define NUM_ITER 5

#define INPUT_N 1
#define INPUT_H 58
#define INPUT_W 58
#define INPUT_C 64
#define KERNEL_H 3
#define KERNEL_W 3
#define KERNEL_C INPUT_C
#define OUTPUT_N INPUT_N
#define OUTPUT_H (INPUT_H - KERNEL_H + 1)
#define OUTPUT_W (INPUT_W - KERNEL_W + 1)
#define OUTPUT_C INPUT_C

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesInput[4] = {INPUT_N, INPUT_H, INPUT_W, INPUT_C};
intptr_t sizesFilter[3] = {KERNEL_H, KERNEL_W, KERNEL_C};
intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_H, OUTPUT_W, OUTPUT_C};

float *inputData = nullptr;
float *filterData = nullptr;

MemRef<float, 4> inputMemRef(sizesInput);
MemRef<float, 3> filterMemRef(sizesFilter);

// Runs the provided Depthwise Conv2D function for benchmarking.
template <typename Func>
void DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC(benchmark::State &state, Func func) {
  MemRef<float, 4> outputMemRef(sizesOutput, 0.0);
  for (auto _ : state) {
    func(&inputMemRef, &filterMemRef, &outputMemRef);
  }
  benchmark::DoNotOptimize(outputMemRef);
}

using MLIRFunctionType = void (*)(MemRef<float, 4> *, MemRef<float, 3> *,
                                  MemRef<float, 4> *);

// Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(float *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<float, 4> outputMemRef(sizesOutput, 0.0);
  MLIRFunc(&inputMemRef, &filterMemRef, &outputMemRef);
  float *outputOptimized = outputMemRef.getData();
  depthwise_conv2d::verify<float>(outputExpected, outputOptimized,
                                  OUTPUT_N * OUTPUT_H * OUTPUT_W * OUTPUT_C,
                                  name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_depthwise_conv_2d_nhwc_hwc_scalar(MemRef<float, 4> *input,
                                                    MemRef<float, 3> *filter,
                                                    MemRef<float, 4> *output);
void _mlir_ciface_depthwise_conv_2d_nhwc_hwc_auto_vectorization(
    MemRef<float, 4> *input, MemRef<float, 3> *filter,
    MemRef<float, 4> *output);
void _mlir_ciface_depthwise_conv_2d_nhwc_hwc_vectorization(
    MemRef<float, 4> *input, MemRef<float, 3> *filter,
    MemRef<float, 4> *output);
/// [Step 1] Add function of your new method here.
}

BENCHMARK_CAPTURE(DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC, scalar,
                  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_scalar)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC, auto_vectorization,
                  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_auto_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC, vectorization,
                  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  inputData =
      depthwise_conv2d::allocArray<float>(INPUT_N, INPUT_H, INPUT_W, INPUT_C);
  filterData =
      depthwise_conv2d::allocArray<float>(KERNEL_H, KERNEL_W, KERNEL_C);

  inputMemRef = MemRef<float, 4>(inputData, sizesInput);
  filterMemRef = MemRef<float, 3>(filterData, sizesFilter);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Obtain scalar output results as expected output for verification.
  MemRef<float, 4> outputMemRefScalar(sizesOutput, 0.0);
  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_scalar(&inputMemRef, &filterMemRef,
                                                 &outputMemRefScalar);
  float *outputExpected = outputMemRefScalar.getData();

  MLIRVerification(outputExpected,
                   _mlir_ciface_depthwise_conv_2d_nhwc_hwc_auto_vectorization,
                   "auto_vectorization");
  MLIRVerification(outputExpected,
                   _mlir_ciface_depthwise_conv_2d_nhwc_hwc_vectorization,
                   "vectorization");
  /// [Step 3] Add your new method for verification.

  delete[] inputData;
  delete[] filterData;
  return 0;
}
