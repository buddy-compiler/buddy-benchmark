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
// This is the main file of MatMulOp benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the number here as needed.
// -----------------------------------------------------------------------------

#define _NUM_ITER 5
#define _SIZE_M 40
#define _SIZE_N 4096
#define _SIZE_K 4096

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesA[2] = {_SIZE_M, _SIZE_K};
intptr_t sizesB[2] = {_SIZE_K, _SIZE_N};
intptr_t sizesC[2] = {_SIZE_M, _SIZE_N};
float *inputA = nullptr;
float *inputB = nullptr;
MemRef<float, 2> inputAMemRef(sizesA);
MemRef<float, 2> inputBMemRef(sizesB);

// Runs the provided MatMul function for benchmarking.
template <typename Func>
void DL_OPS_MATMUL(benchmark::State &state, Func func) {
  MemRef<float, 2> outputMemRef(sizesC, 0.0);
  for (auto _ : state) {
    func(&inputAMemRef, &inputBMemRef, &outputMemRef);
  }
  benchmark::DoNotOptimize(outputMemRef);
}

using MLIRFunctionType = void (*)(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
//  Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(float *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<float, 2> outputMemRef(sizesC, 0);
  MLIRFunc(&inputAMemRef, &inputBMemRef, &outputMemRef);
  float *outputOptimized = outputMemRef.getData();
  matmul::verify<float>(outputExpected, outputOptimized, _SIZE_M, _SIZE_N,
                        name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_matmul_scalar_O0(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
void _mlir_ciface_matmul_scalar_O3(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
void _mlir_ciface_matmul_tile(MemRef<float, 2> *A, MemRef<float, 2> *B,
                              MemRef<float, 2> *C);
void _mlir_ciface_matmul_vec(MemRef<float, 2> *A, MemRef<float, 2> *B,
                             MemRef<float, 2> *C);
void _mlir_ciface_matmul_vec_omp(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                 MemRef<float, 2> *C);
/// [Step 1] Add function of your new method.
}
BENCHMARK_CAPTURE(DL_OPS_MATMUL, scalar_O0, _mlir_ciface_matmul_scalar_O0)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL, scalar_O3, _mlir_ciface_matmul_scalar_O3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL, tile, _mlir_ciface_matmul_tile)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL, vec, _mlir_ciface_matmul_vec)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL, vec_omp, _mlir_ciface_matmul_vec_omp)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  inputA = matmul::allocArray<float>(_SIZE_N, _SIZE_M);
  inputB = matmul::allocArray<float>(_SIZE_N, _SIZE_M);
  inputAMemRef = MemRef<float, 2>(inputA, sizesA);
  inputBMemRef = MemRef<float, 2>(inputB, sizesB);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Attain scalar output results as expected output results in verification.
  MemRef<float, 2> outputMemrefScalar(sizesC, 0);
  _mlir_ciface_matmul_scalar_O3(&inputAMemRef, &inputBMemRef,
                                &outputMemrefScalar);
  float *outputExpected = outputMemrefScalar.getData();

  MLIRVerification(outputExpected, _mlir_ciface_matmul_tile, "tile");
  MLIRVerification(outputExpected, _mlir_ciface_matmul_vec, "vec");
  MLIRVerification(outputExpected, _mlir_ciface_matmul_vec_omp, "vec_omp");
  /// [Step 3] Add your new method for verification.

  delete[] inputA;
  delete[] inputB;
  return 0;
}
