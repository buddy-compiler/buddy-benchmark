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
// This is the main file of Batch MatMul benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the number here as needed.
// -----------------------------------------------------------------------------

#define BATCH_SIZE 3
#define _SIZE_M (128 * 4)
#define _SIZE_N (128 * 4)
#define _SIZE_K (128 * 4)
#define _NUM_ITER 5

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesInput1[3] = {BATCH_SIZE, _SIZE_M, _SIZE_K};
intptr_t sizesInput2[3] = {BATCH_SIZE, _SIZE_K, _SIZE_N};
intptr_t sizesOutput[3] = {BATCH_SIZE, _SIZE_M, _SIZE_N};
float *input1 = nullptr;
float *input2 = nullptr;
MemRef<float, 3> input1MemRef(sizesInput1);
MemRef<float, 3> input2MemRef(sizesInput2);

// Runs the provided BatchMatMul function for benchmarking.
template <typename Func>
void DL_OPS_BATCH_MATMUL(benchmark::State &state, Func func) {
  MemRef<float, 3> outputMemRef(sizesOutput, 0.0);
  for (auto _ : state) {
    func(&input1MemRef, &input2MemRef, &outputMemRef);
  }
  benchmark::DoNotOptimize(outputMemRef);
}

using MLIRFunctionType = void (*)(MemRef<float, 3> *, MemRef<float, 3> *,
                                  MemRef<float, 3> *);
// Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(float *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<float, 3> outputMemRef(sizesOutput, 0);
  MLIRFunc(&input1MemRef, &input2MemRef, &outputMemRef);
  float *outputOptimized = outputMemRef.getData();
  batch_matmul_int::verify<float>(outputExpected, outputOptimized, BATCH_SIZE,
                                  _SIZE_M * _SIZE_N, name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_batch_matmul_scalar(MemRef<float, 3> *A, MemRef<float, 3> *B,
                                      MemRef<float, 3> *C);
void _mlir_ciface_batch_matmul_auto_vectorization(MemRef<float, 3> *A,
                                                  MemRef<float, 3> *B,
                                                  MemRef<float, 3> *C);
void _mlir_ciface_batch_matmul_vectorization(MemRef<float, 3> *A,
                                             MemRef<float, 3> *B,
                                             MemRef<float, 3> *C);
void _mlir_ciface_batch_matmul_rvv_vectorization(MemRef<float, 3> *A,
                                                 MemRef<float, 3> *B,
                                                 MemRef<float, 3> *C);
/// [Step 1] Add function of your new method.
}

BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL, Scalar, _mlir_ciface_batch_matmul_scalar)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL, AutoVectorization,
                  _mlir_ciface_batch_matmul_auto_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL, Vectorization,
                  _mlir_ciface_batch_matmul_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL, RVVVectorization,
                  _mlir_ciface_batch_matmul_rvv_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  input1 = batch_matmul_int::allocArray<float>(BATCH_SIZE * _SIZE_M, _SIZE_K);
  input2 = batch_matmul_int::allocArray<float>(BATCH_SIZE * _SIZE_K, _SIZE_N);
  input1MemRef = MemRef<float, 3>(input1, sizesInput1);
  input2MemRef = MemRef<float, 3>(input2, sizesInput2);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Attain scalar output results as expected output results in verification.
  MemRef<float, 3> outputMemrefScalar(sizesOutput, 0);
  _mlir_ciface_batch_matmul_scalar(&input1MemRef, &input2MemRef,
                                   &outputMemrefScalar);
  float *outputExpected = outputMemrefScalar.getData();

  MLIRVerification(outputExpected, _mlir_ciface_batch_matmul_auto_vectorization,
                   "AutoVectorization");
  MLIRVerification(outputExpected, _mlir_ciface_batch_matmul_vectorization,
                   "Vectorization");
  MLIRVerification(outputExpected, _mlir_ciface_batch_matmul_rvv_vectorization,
                   "RVVVectorization");
  /// [Step 3] Add your new method for verification.

  delete[] input1;
  delete[] input2;
  return 0;
}
