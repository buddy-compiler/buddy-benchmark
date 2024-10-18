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
// This is the main file of the Conv2D NHWC-FHWC benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the numbers here as needed.
// -----------------------------------------------------------------------------

#define NUM_ITER 5
#define INPUT_N 2
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_C 18
#define FILTER_F 66
#define FILTER_H 4
#define FILTER_W 4
#define FILTER_C INPUT_C
#define OUTPUT_N INPUT_N
#define OUTPUT_H (INPUT_H - FILTER_H + 1)
#define OUTPUT_W (INPUT_W - FILTER_W + 1)
#define OUTPUT_F FILTER_F

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesInput[4] = {INPUT_N, INPUT_H, INPUT_W, INPUT_C};
intptr_t sizesFilter[4] = {FILTER_F, FILTER_H, FILTER_W, FILTER_C};
intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_H, OUTPUT_W, OUTPUT_F};

int *inputData = nullptr;
int *filterData = nullptr;

MemRef<int, 4> inputMemRef(sizesInput);
MemRef<int, 4> filterMemRef(sizesFilter);

// Runs the provided Conv2D function for benchmarking.
template <typename Func>
void DL_OPS_CONV_2D_NHWC_FHWC(benchmark::State &state, Func func) {
  MemRef<int, 4> outputMemRef(sizesOutput, 0);
  for (auto _ : state) {
    func(&inputMemRef, &filterMemRef, &outputMemRef);
  }
  benchmark::DoNotOptimize(outputMemRef);
}

using MLIRFunctionType = void (*)(MemRef<int, 4> *, MemRef<int, 4> *,
                                  MemRef<int, 4> *);

// Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(int *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<int, 4> outputMemRef(sizesOutput, 0);
  MLIRFunc(&inputMemRef, &filterMemRef, &outputMemRef);
  int *outputOptimized = outputMemRef.getData();
  conv2d_nhwc_fhwc::verify<int>(outputExpected, outputOptimized,
                                OUTPUT_N * OUTPUT_H * OUTPUT_W * OUTPUT_F,
                                name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_conv_2d_nhwc_fhwc_scalar(MemRef<int, 4> *input,
                                           MemRef<int, 4> *filter,
                                           MemRef<int, 4> *output);
void _mlir_ciface_conv_2d_nhwc_fhwc_auto_vectorization(MemRef<int, 4> *input,
                                                       MemRef<int, 4> *filter,
                                                       MemRef<int, 4> *output);
void _mlir_ciface_conv_2d_nhwc_fhwc_vectorization(MemRef<int, 4> *input,
                                                  MemRef<int, 4> *filter,
                                                  MemRef<int, 4> *output);
void _mlir_ciface_conv_2d_nhwc_fhwc_rvv_vectorization(MemRef<int, 4> *input,
                                                      MemRef<int, 4> *filter,
                                                      MemRef<int, 4> *output);
/// [Step 1] Add function of your new method here.
}

BENCHMARK_CAPTURE(DL_OPS_CONV_2D_NHWC_FHWC, scalar,
                  _mlir_ciface_conv_2d_nhwc_fhwc_scalar)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_CONV_2D_NHWC_FHWC, auto_vectorization,
                  _mlir_ciface_conv_2d_nhwc_fhwc_auto_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_CONV_2D_NHWC_FHWC, vectorization,
                  _mlir_ciface_conv_2d_nhwc_fhwc_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_CONV_2D_NHWC_FHWC, rvv_vectorization,
                  _mlir_ciface_conv_2d_nhwc_fhwc_rvv_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  inputData =
      conv2d_nhwc_fhwc::allocArray<int>(INPUT_N, INPUT_H, INPUT_W, INPUT_C);
  filterData =
      conv2d_nhwc_fhwc::allocArray<int>(FILTER_F, FILTER_H, FILTER_W, FILTER_C);

  inputMemRef = MemRef<int, 4>(inputData, sizesInput);
  filterMemRef = MemRef<int, 4>(filterData, sizesFilter);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Obtain scalar output results as expected output for verification.
  MemRef<int, 4> outputMemRefScalar(sizesOutput, 0);
  _mlir_ciface_conv_2d_nhwc_fhwc_scalar(&inputMemRef, &filterMemRef,
                                        &outputMemRefScalar);
  int *outputExpected = outputMemRefScalar.getData();

  MLIRVerification(outputExpected,
                   _mlir_ciface_conv_2d_nhwc_fhwc_auto_vectorization,
                   "auto_vectorization");
  MLIRVerification(outputExpected, _mlir_ciface_conv_2d_nhwc_fhwc_vectorization,
                   "vectorization");
  MLIRVerification(outputExpected,
                   _mlir_ciface_conv_2d_nhwc_fhwc_rvv_vectorization,
                   "rvv_vectorization");
  /// [Step 3] Add your new method for verification.

  delete[] inputData;
  delete[] filterData;
  return 0;
}
