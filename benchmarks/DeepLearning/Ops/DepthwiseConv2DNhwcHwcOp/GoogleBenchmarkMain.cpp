//===- GoogleBenchmarkMain.cpp --------------------------------------------===//
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

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#define INPUT_N 1
#define INPUT_C 64
#define INPUT_H 58
#define INPUT_W 58
#define KERNEL_C INPUT_C
#define KERNEL_H 3
#define KERNEL_W 3
#define OUTPUT_N INPUT_N
#define OUTPUT_C INPUT_C
#define OUTPUT_H (INPUT_H - KERNEL_H + 1)
#define OUTPUT_W (INPUT_W - KERNEL_W + 1)

// Helper functions and variables.
namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(float array1[], float array2[], int size,
                    float epsilon = 0.0001) {
  for (int i = 0; i < size; ++i) {
    if (fabs(array1[i] - array2[i]) > epsilon) {
      printf("array1[%d]:%f array2[%d]:%f\n", i, array1[i], i, array2[i]);
      return false;
    }
  }
  return true;
}

} // namespace

namespace {

// Declare the mobilenet C interface.
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
}

template <typename Func>
void DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC(benchmark::State &state, Func func) {

  intptr_t sizesInput[4] = {INPUT_N, INPUT_H, INPUT_W, INPUT_C};
  intptr_t sizesKernel[3] = {KERNEL_H, KERNEL_W, KERNEL_C};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_H, OUTPUT_W, OUTPUT_C};

  MemRef<float, 4> inputMemRef(sizesInput, 2.0);
  MemRef<float, 3> filterMemRef(sizesKernel, 3.0);
  MemRef<float, 4> outputMemRef(sizesOutput, 0.0);

  for (auto _ : state) {
    func(&inputMemRef, &filterMemRef, &outputMemRef);
  }
}
} // namespace

// Register benchmarking function with different arguments.
BENCHMARK_CAPTURE(DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC, Scalar,
                  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_scalar)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC, AutoVectorization,
                  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_auto_vectorization)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC, Vectorization,
                  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_vectorization)
    ->Unit(benchmark::kMillisecond);
// BENCHMARK_CAPTURE(DL_OPS_DEPTHWISE_CONV_2D_NHWC_HWC, RvvVectorization,
//                   _mlir_ciface_depthwise_conv_2d_nhwc_hwc_rvv_vectorization)
//     ->Unit(benchmark::kMillisecond);

/// Correctness Verification
/// The verification does not affect the performance.
/// - Set the scalar case as the criteria.
/// - Input elements are random numbers.
/// - Output elements are initialized to zero.
/// - Compare the output of various optimizations with the scalar version to
///   verify correctness.
void verification() {
  // Set the random number generator.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  // Set the layout sizes of input and kernel memref container.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_H, INPUT_W, INPUT_C};
  intptr_t sizesKernel[4] = {KERNEL_H, KERNEL_W, KERNEL_C};
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_H, OUTPUT_W, OUTPUT_C};

  // Generate input memref container with random numbers.
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;
  const int outputSize = INPUT_N * OUTPUT_C * OUTPUT_H * OUTPUT_W;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }
  MemRef<float, 4> inputMemRef(inputRand, sizesInput);

  // Generate kernel memref container with random numbers.
  const int kernelSize = KERNEL_C * KERNEL_H * KERNEL_W;
  float kernelRand[kernelSize];
  for (int i = 0; i < kernelSize; ++i) {
    kernelRand[i] = distribution(generator);
  }
  MemRef<float, 3> filterMemRef(kernelRand, sizesKernel);

  // Generate output memref container with zero.
  MemRef<float, 4> outputScalar(sizesOutput, 0.0);
  MemRef<float, 4> outputAutoVectorization(sizesOutput, 0.0);
  MemRef<float, 4> outputVectorization(sizesOutput, 0.0);
  // MemRef<int, 4> outputRVVVectorization(sizesOutput, 0.0);

  // Perform all the matmul implementation.
  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_scalar(&inputMemRef, &filterMemRef,
                                                 &outputScalar);
  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_auto_vectorization(
      &inputMemRef, &filterMemRef, &outputAutoVectorization);
  _mlir_ciface_depthwise_conv_2d_nhwc_hwc_vectorization(
      &inputMemRef, &filterMemRef, &outputVectorization);
  // _mlir_ciface_depthwise_conv_2d_nhwc_hwc_rvv_vectorization(
  //     &inputMemRef, &filterMemRef, &outputRVVVectorization);

  // Get the result array.
  auto resultScalar = outputScalar.getData();
  auto resultAutoVectorization = outputAutoVectorization.getData();
  auto resultVectorization = outputVectorization.getData();
  // auto resultRVVVectorization = outputRVVVectorization.getData();

  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Scalar vs AutoVectorization: "
            << (areArraysEqual(resultScalar, resultAutoVectorization,
                               outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  std::cout << "Scalar vs Vectorization: "
            << (areArraysEqual(resultScalar, resultVectorization, outputSize)
                    ? PASS
                    : FAIL)
            << std::endl;
  // std::cout << "Scalar vs AutoTillingVectorization: "
  //           << (areArraysEqual(resultScalar, resultRVVVectorization,
  //           outputSize)
  //                   ? PASS
  //                   : FAIL)
  //           << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
}

int main(int argc, char **argv) {
  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  // Run correctness verification.
  verification();
  return 0;
}
