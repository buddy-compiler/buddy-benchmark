//===----------------------------- Main.cpp -------------------------------===//
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
// This is the main file of MatMulOp benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>

// -----------------------------------------------------------------------------
// Benchmark Configuration
// -----------------------------------------------------------------------------

#define _SIZE_M 16
#define _SIZE_N 3136
#define _SIZE_K 576
#define _NUM_ITER 5

float *inputA = nullptr;
float *inputB = nullptr;

// -----------------------------------------------------------------------------
// MLIR Scalar Benchmark
// -----------------------------------------------------------------------------

template <typename Func>
void DL_OPS_MATMUL(benchmark::State &state, Func func) {
  intptr_t sizesA[2] = {_SIZE_M, _SIZE_K};
  intptr_t sizesB[2] = {_SIZE_K, _SIZE_N};
  intptr_t sizesC[2] = {_SIZE_M, _SIZE_N};
  MemRef<float, 2> A(sizesA, 1.0);
  MemRef<float, 2> B(sizesB, 1.0);
  MemRef<float, 2> C(sizesC, 0.0);
  for (auto _ : state) {
    func(&A, &B, &C);
  }
  benchmark::DoNotOptimize(C);
}
extern "C" {
void _mlir_ciface_matmul_scalar_O0(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
void _mlir_ciface_matmul_scalar(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                MemRef<float, 2> *C);
}
BENCHMARK_CAPTURE(DL_OPS_MATMUL, scalar_O0, _mlir_ciface_matmul_scalar_O0)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL, scalar, _mlir_ciface_matmul_scalar)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);

// -----------------------------------------------------------------------------
// MLIR Vector Benchmark
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_matmul_fixed_vector_scalar(MemRef<float, 2> *A,
                                             MemRef<float, 2> *B,
                                             MemRef<float, 2> *C);
void _mlir_ciface_matmul_fixed_vector_mask(MemRef<float, 2> *A,
                                           MemRef<float, 2> *B,
                                           MemRef<float, 2> *C);
}
BENCHMARK_CAPTURE(DL_OPS_MATMUL, fixed_vector_scalar,
                  _mlir_ciface_matmul_fixed_vector_scalar)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL, fixed_vector_mask,
                  _mlir_ciface_matmul_fixed_vector_mask)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
/// [Step 1] Add your new case for benchmarking.

// -----------------------------------------------------------------------------
// Verification Functions
// -----------------------------------------------------------------------------

using MLIRFunctionType = void (*)(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
template <typename DATA_TYPE>
void MLIRVerification(MLIRFunctionType MLIRFunc, const std::string &name,
                      MemRef<DATA_TYPE, 2> &inputAMemRef,
                      MemRef<DATA_TYPE, 2> &inputBMemRef,
                      DATA_TYPE *outputScalar, int rows, int cols) {
  intptr_t sizesC[2] = {rows, cols};
  MemRef<DATA_TYPE, 2> outputMemRef(sizesC, 0);
  MLIRFunc(&inputAMemRef, &inputBMemRef, &outputMemRef);
  DATA_TYPE *outputOptimized = outputMemRef.getData();
  matmul::verify<DATA_TYPE>(name, outputScalar, outputOptimized, rows, cols);
}

// -----------------------------------------------------------------------------
// Main Function
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Run correctness verification of optimized cases.
  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;

  // Initialize input data.
  intptr_t sizesA[2] = {_SIZE_M, _SIZE_K};
  intptr_t sizesB[2] = {_SIZE_K, _SIZE_N};
  intptr_t sizesC[2] = {_SIZE_M, _SIZE_N};
  inputA = matmul::allocArray<float>(_SIZE_N, _SIZE_M);
  inputB = matmul::allocArray<float>(_SIZE_N, _SIZE_M);
  MemRef<float, 2> inputAMemRef(inputA, sizesA);
  MemRef<float, 2> inputBMemRef(inputB, sizesB);

  // Attain scalar output results.
  MemRef<float, 2> outputMemrefScalar(sizesC, 0);
  _mlir_ciface_matmul_scalar(&inputAMemRef, &inputBMemRef, &outputMemrefScalar);
  float *outputScalar = outputMemrefScalar.getData();

  MLIRVerification<float>(_mlir_ciface_matmul_fixed_vector_scalar,
                          "fixed_vector_scalar", inputAMemRef, inputBMemRef,
                          outputScalar, _SIZE_M, _SIZE_N);
  MLIRVerification<float>(_mlir_ciface_matmul_fixed_vector_mask,
                          "fixed_vector_mask", inputAMemRef, inputBMemRef,
                          outputScalar, _SIZE_M, _SIZE_N);
  /// [Step 2] Add your new case for evaluation.

  delete[] inputA;
  delete[] inputB;
  return 0;
}
