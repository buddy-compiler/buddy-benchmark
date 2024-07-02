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
// This file implements the benchmark for conv2d(nchw-fchw) operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>

#define INPUT_N 1
#define INPUT_C 64
#define INPUT_H 58
#define INPUT_W 58
#define KERNEL_F 64
#define KERNEL_C INPUT_C
#define KERNEL_H 3
#define KERNEL_W 3
#define OUTPUT_N INPUT_N
#define OUTPUT_C KERNEL_F
#define OUTPUT_H (INPUT_H - KERNEL_H + 1)
#define OUTPUT_W (INPUT_W - KERNEL_W + 1)

// Helper functions and variables.
namespace {
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[31mFAIL\033[0m";

bool areArraysEqual(float array1[], float array2[], int size, float epsilon = 0.0001) {
    for (int i = 0; i < size; ++i) {
        if (fabs(array1[i] - array2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

// The printResult function iterates through the output array organized in NCHW format
// (N, C, H, and W), prints the values for each batch, channel, row, and column.
template <size_t N, size_t C, size_t H, size_t W, typename T>
void printResult(T output) {
    // Iterate over each batch
    for (size_t n = 0; n < N; ++n) {
        std::cout << "N " << n + 1 << ":\n";
        // Iterate over each channel
        for (size_t c = 0; c < C; ++c) {
            std::cout << "Channel " << c + 1 << ":\n";
            // Iterate over each row
            for (size_t h = 0; h < H; ++h) {
                // Iterate over each column
                for (size_t w = 0; w < W; ++w) {
                    // Calculate the index for the output array based on NCHW format
                    size_t index = n * C * H * W + c * H * W + h * W + w;
                    std::cout << std::fixed << std::setprecision(4) << output[index] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

} // namespace


namespace {

// Declare the mobilenet C interface.
extern "C" {
void _mlir_ciface_conv_2d_nchw_fchw_scalar(MemRef<float, 4> *input,
                                    MemRef<float, 4> *filter,
                                    MemRef<float, 4> *output);

void _mlir_ciface_conv_2d_nchw_fchw_im2col(MemRef<float, 4> *input,
                                    MemRef<float, 4> *filter,
                                    MemRef<float, 4> *output);
}

#define DEFINE_Conv2DNchwFchw_BENCHMARK(name, func)                            \
  void BM_Conv2DNchwFchw_##name(benchmark::State &state) {                     \
    intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};             \
    intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};        \
    intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_C, OUTPUT_H, OUTPUT_W};        \
                                                                               \
    MemRef<float, 4> inputMemRef(sizesInput, 2.0);                             \
    MemRef<float, 4> filterMemRef(sizesKernel, 3.0);                           \
    MemRef<float, 4> outputMemRef(sizesOutput, 0.0);                           \
                                                                               \
    for (auto _ : state) {                                                     \
      func(&inputMemRef, &filterMemRef, &outputMemRef);                        \
    }                                                                          \
  }                                                        

DEFINE_Conv2DNchwFchw_BENCHMARK(SCALAR, _mlir_ciface_conv_2d_nchw_fchw_scalar)
DEFINE_Conv2DNchwFchw_BENCHMARK(Im2col, _mlir_ciface_conv_2d_nchw_fchw_im2col)

} // namespace

// Register benchmarking function with different arguments.
BENCHMARK(BM_Conv2DNchwFchw_SCALAR)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Conv2DNchwFchw_Im2col)->Unit(benchmark::kMillisecond);

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
  // std::uniform_int_distribution<int> distribution(1, 100);

  // Set the layout sizes of input and kernel memref container.
  intptr_t sizesInput[4] = {INPUT_N, INPUT_C, INPUT_H, INPUT_W};                                
  intptr_t sizesKernel[4] = {KERNEL_F, KERNEL_C, KERNEL_H, KERNEL_W};                                  
  intptr_t sizesOutput[4] = {OUTPUT_N, OUTPUT_C, OUTPUT_H, OUTPUT_W};                               

  // Generate input memref container with random numbers.                                                     
  const int inputSize = INPUT_N * INPUT_C * INPUT_H * INPUT_W;
  float inputRand[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputRand[i] = distribution(generator);
  }
  MemRef<float, 4> inputMemRef(inputRand, sizesInput);

  // Generate kernel memref container with random numbers.   
  const int kernelSize = KERNEL_F * KERNEL_C * KERNEL_H * KERNEL_W;
  float kernelRand[kernelSize];
  for (int i = 0; i < kernelSize; ++i) {
    kernelRand[i] = distribution(generator);
  }
  MemRef<float, 4> filterMemRef(kernelRand, sizesKernel);  

  // Generate output memref container with zero.
  MemRef<float, 4> outputMemRef(sizesOutput, 0.0); 
  MemRef<float, 4> outputTransform(sizesOutput, 0.0);

  // Perform all the matmul implementation.
  _mlir_ciface_conv_2d_nchw_fchw_scalar(&inputMemRef, &filterMemRef, &outputMemRef);   
  _mlir_ciface_conv_2d_nchw_fchw_im2col(&inputMemRef, &filterMemRef, &outputTransform);

  
  // Get the result array.
  auto resultScalar = outputMemRef.getData();
  auto resultTransform = outputTransform.getData();

  // printResult<OUTPUT_N, OUTPUT_C, OUTPUT_H, OUTPUT_W>(resultScalar);

  // Print the verfication result.
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  std::cout << "Transform case: "
            << (areArraysEqual(resultScalar, resultTransform, OUTPUT_N * OUTPUT_C * OUTPUT_H * OUTPUT_W)
                    ? PASS
                    : FAIL)
            << std::endl;
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



