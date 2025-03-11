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
// This is the main file of Batch Matmul TransposeBOp benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the number here as needed.
// -----------------------------------------------------------------------------

#define _NUM_ITER 1
#define _SIZE_BATCH 4
#define _SIZE_N 40
#define _SIZE_K 256
#define _SIZE_M 256

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesInput1[3] = {_SIZE_BATCH, _SIZE_M, _SIZE_K};
intptr_t sizesInput2[3] = {_SIZE_BATCH, _SIZE_K, _SIZE_N};
intptr_t sizesOutput[3] = {_SIZE_BATCH, _SIZE_M, _SIZE_N};
float *input1 = nullptr;
float *input2 = nullptr;
MemRef<float, 3> input1MemRef(sizesInput1);
MemRef<float, 3> input2MemRef(sizesInput2);

// Runs the provided BatchMatMulTransposeB function for benchmarking.
template <typename Func>
void DL_OPS_BATCH_MATMUL_TRANSPOSE_B(benchmark::State &state, Func func) {
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
  batch_matmul_transpose_b::verify<float>(outputExpected, outputOptimized,
                                          _SIZE_BATCH, _SIZE_M * _SIZE_N, name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_batch_matmul_transpose_b_scalar_O0(MemRef<float, 3> *A,
                                                     MemRef<float, 3> *B,
                                                     MemRef<float, 3> *C);
void _mlir_ciface_batch_matmul_transpose_b_scalar_O3(MemRef<float, 3> *A,
                                                     MemRef<float, 3> *B,
                                                     MemRef<float, 3> *C);
void _mlir_ciface_batch_matmul_transpose_b_vec(MemRef<float, 3> *A,
                                               MemRef<float, 3> *B,
                                               MemRef<float, 3> *C);
/// [Step 1] Add function of your new method.
}
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL_TRANSPOSE_B, Scalar_O0,
                  _mlir_ciface_batch_matmul_transpose_b_scalar_O0)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL_TRANSPOSE_B, Scalar_O3,
                  _mlir_ciface_batch_matmul_transpose_b_scalar_O3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_BATCH_MATMUL_TRANSPOSE_B, Vec,
                  _mlir_ciface_batch_matmul_transpose_b_vec)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);

/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  input1 = batch_matmul_transpose_b::allocArray<float>(_SIZE_BATCH * _SIZE_N,
                                                       _SIZE_K);
  input2 = batch_matmul_transpose_b::allocArray<float>(_SIZE_BATCH * _SIZE_K,
                                                       _SIZE_M);
  input1MemRef = MemRef<float, 3>(input1, sizesInput1);
  input2MemRef = MemRef<float, 3>(input2, sizesInput2);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Attain scalar output results as expected output results in verification.
  MemRef<float, 3> outputMemrefScalar(sizesOutput, 0);
  _mlir_ciface_batch_matmul_transpose_b_scalar_O0(&input1MemRef, &input2MemRef,
                                                  &outputMemrefScalar);
  float *outputExpected = outputMemrefScalar.getData();

  MLIRVerification(outputExpected,
                   _mlir_ciface_batch_matmul_transpose_b_scalar_O3,
                   "Scalar_O3");
  MLIRVerification(outputExpected, _mlir_ciface_batch_matmul_transpose_b_vec,
                   "Vec");
  /// [Step 3] Add your new method for verification.

  delete[] input1;
  delete[] input2;
  return 0;
}
