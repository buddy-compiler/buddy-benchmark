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
// This is the main file of the MatMul benchmark.
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
#define SIZE_M (128 * 4)
#define SIZE_N (128 * 4)
#define SIZE_K (128 * 4)

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesA[2] = {SIZE_M, SIZE_K};
intptr_t sizesB[2] = {SIZE_K, SIZE_N};
intptr_t sizesC[2] = {SIZE_M, SIZE_N};

int *inputA = nullptr;
int *inputB = nullptr;

MemRef<int, 2> inputAMemRef(sizesA);
MemRef<int, 2> inputBMemRef(sizesB);

// Runs the provided MatMul function for benchmarking.
template <typename Func>
void DL_OPS_MATMUL(benchmark::State &state, Func func) {
  MemRef<int, 2> outputMemRef(sizesC, 0);
  for (auto _ : state) {
    func(&inputAMemRef, &inputBMemRef, &outputMemRef);
  }
  benchmark::DoNotOptimize(outputMemRef);
}

using MLIRFunctionType = void (*)(MemRef<int, 2> *, MemRef<int, 2> *,
                                  MemRef<int, 2> *);

// Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(int *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<int, 2> outputMemRef(sizesC, 0);
  MLIRFunc(&inputAMemRef, &inputBMemRef, &outputMemRef);
  int *outputOptimized = outputMemRef.getData();
  matmul::verify<int>(outputExpected, outputOptimized, SIZE_M, SIZE_N, name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_matmul_scalar(MemRef<int, 2> *A, MemRef<int, 2> *B,
                                MemRef<int, 2> *C);
void _mlir_ciface_matmul_vec(MemRef<int, 2> *A, MemRef<int, 2> *B,
                             MemRef<int, 2> *C);
void _mlir_ciface_matmul_rvv(MemRef<int, 2> *A, MemRef<int, 2> *B,
                             MemRef<int, 2> *C);
/// [Step 1] Add function of your new method here.
}

BENCHMARK_CAPTURE(DL_OPS_MATMUL, scalar, _mlir_ciface_matmul_scalar)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL, vec, _mlir_ciface_matmul_vec)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL, rvv, _mlir_ciface_matmul_rvv)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITER);
/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  inputA = matmul::allocArray<int>(SIZE_M, SIZE_K);
  inputB = matmul::allocArray<int>(SIZE_K, SIZE_N);

  inputAMemRef = MemRef<int, 2>(inputA, sizesA);
  inputBMemRef = MemRef<int, 2>(inputB, sizesB);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Obtain scalar output results as expected output for verification.
  MemRef<int, 2> outputMemRefScalar(sizesC, 0);
  _mlir_ciface_matmul_scalar(&inputAMemRef, &inputBMemRef, &outputMemRefScalar);
  int *outputExpected = outputMemRefScalar.getData();

  MLIRVerification(outputExpected, _mlir_ciface_matmul_vec, "vec");
  MLIRVerification(outputExpected, _mlir_ciface_matmul_rvv, "rvv");
  /// [Step 3] Add your new method for verification.

  delete[] inputA;
  delete[] inputB;
  return 0;
}
