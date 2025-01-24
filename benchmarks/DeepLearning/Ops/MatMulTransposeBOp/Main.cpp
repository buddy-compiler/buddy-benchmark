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
// This is the main file of MatmulTransposeBOp benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the number here as needed.
// -----------------------------------------------------------------------------

#define _NUM_ITER 5
#define _SIZE_A_ROW 40
#define _SIZE_A_COL 4096
#define _SIZE_B_ROW 4096
#define _SIZE_B_COL 4096
#define _SIZE_C_ROW 40
#define _SIZE_C_COL 4096

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesA[2] = {_SIZE_A_ROW, _SIZE_A_COL};
intptr_t sizesB[2] = {_SIZE_B_ROW, _SIZE_B_COL};
intptr_t sizesC[2] = {_SIZE_C_ROW, _SIZE_C_COL};
float *dataA = nullptr;
float *dataB = nullptr;
MemRef<float, 2> inputAMemRef(sizesA);
MemRef<float, 2> inputBMemRef(sizesB);

// Runs the provided matmul_transpose_b function for benchmarking.
template <typename Func>
void DL_OPS_MATMUL_TRANSPOSE_B(benchmark::State &state, Func func) {
  MemRef<float, 2> outputMemRef(sizesC, 0.0);
  for (auto _ : state) {
    func(&outputMemRef, &inputAMemRef, &inputBMemRef);
  }
  benchmark::DoNotOptimize(outputMemRef);
}

using MLIRFunctionType = void (*)(MemRef<float, 2> *, MemRef<float, 2> *,
                                  MemRef<float, 2> *);
//  Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(float *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<float, 2> outputMemRef(sizesC, 0);
  MLIRFunc(&outputMemRef, &inputAMemRef, &inputBMemRef);
  float *outputOptimized = outputMemRef.getData();
  matmul_transpose_b::verify<float>(outputExpected, outputOptimized,
                                    _SIZE_C_ROW, _SIZE_C_COL, name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_matmul_transpose_b_scalar_O0(MemRef<float, 2> *C,
                                               MemRef<float, 2> *A,
                                               MemRef<float, 2> *B);
void _mlir_ciface_matmul_transpose_b_scalar_O3(MemRef<float, 2> *C,
                                               MemRef<float, 2> *A,
                                               MemRef<float, 2> *B);
void _mlir_ciface_matmul_transpose_b_scalar_O3_omp(MemRef<float, 2> *C,
                                                   MemRef<float, 2> *A,
                                                   MemRef<float, 2> *B);
void _mlir_ciface_matmul_transpose_b_vec(MemRef<float, 2> *C,
                                         MemRef<float, 2> *A,
                                         MemRef<float, 2> *B);
/// [Step 1] Add function of your new method.
}
BENCHMARK_CAPTURE(DL_OPS_MATMUL_TRANSPOSE_B, scalar_O0,
                  _mlir_ciface_matmul_transpose_b_scalar_O0)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL_TRANSPOSE_B, scalar_O3,
                  _mlir_ciface_matmul_transpose_b_scalar_O3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL_TRANSPOSE_B, scalar_O3_omp,
                  _mlir_ciface_matmul_transpose_b_scalar_O3_omp)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_MATMUL_TRANSPOSE_B, vec,
                  _mlir_ciface_matmul_transpose_b_vec)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);

/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  dataA = matmul_transpose_b::allocArray<float>(_SIZE_A_ROW, _SIZE_A_COL);
  dataB = matmul_transpose_b::allocArray<float>(_SIZE_B_ROW, _SIZE_B_COL);
  inputAMemRef = MemRef<float, 2>(dataA, sizesA);
  inputBMemRef = MemRef<float, 2>(dataB, sizesB);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Attain scalar output results as expected output results in verification.
  MemRef<float, 2> outputMemrefScalar(sizesC, 0);
  _mlir_ciface_matmul_transpose_b_scalar_O0(&outputMemrefScalar, &inputAMemRef,
                                            &inputBMemRef);
  float *outputExpected = outputMemrefScalar.getData();

  MLIRVerification(outputExpected, _mlir_ciface_matmul_transpose_b_scalar_O3,
                   "scalar_O3");
  MLIRVerification(outputExpected,
                   _mlir_ciface_matmul_transpose_b_scalar_O3_omp,
                   "scalar_O3_omp");
  MLIRVerification(outputExpected, _mlir_ciface_matmul_transpose_b_vec, "vec");
  /// [Step 3] Add your new method for verification.

  delete[] dataA;
  delete[] dataB;
  return 0;
}
