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
// This is the main file of TransposeOp benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the number here as needed.
// -----------------------------------------------------------------------------

#define _NUM_ITER 5
#define _SIZE_ROW 4096
#define _SIZE_COL 4096

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

intptr_t sizesInput2D[2] = {_SIZE_ROW, _SIZE_COL};
intptr_t sizesOutput2D[2] = {_SIZE_COL, _SIZE_ROW};
float *dataInput2D = nullptr;
MemRef<float, 2> inputMemRef2D(sizesInput2D);

// Runs the provided Transpose function for benchmarking.
template <typename Func>
void DL_OPS_TRANSPOSE_2D(benchmark::State &state, Func func) {
  MemRef<float, 2> outputMemRef2D(sizesOutput2D, 0.0);
  for (auto _ : state) {
    func(&outputMemRef2D, &inputMemRef2D);
  }
  benchmark::DoNotOptimize(outputMemRef2D);
}

using MLIRFunctionType = void (*)(MemRef<float, 2> *, MemRef<float, 2> *);
//  Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(float *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<float, 2> outputMemRef(sizesOutput2D, 0);
  MLIRFunc(&outputMemRef, &inputMemRef2D);
  float *outputOptimized = outputMemRef.getData();
  transpose::verify<float>(outputExpected, outputOptimized, _SIZE_COL,
                           _SIZE_ROW, name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_transpose_2d_scalar_O0(MemRef<float, 2> *A,
                                         MemRef<float, 2> *B);
void _mlir_ciface_transpose_2d_scalar_O3(MemRef<float, 2> *A,
                                         MemRef<float, 2> *B);
/// [Step 1] Add function of your new method.
}
BENCHMARK_CAPTURE(DL_OPS_TRANSPOSE_2D, scalar_O0,
                  _mlir_ciface_transpose_2d_scalar_O0)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DL_OPS_TRANSPOSE_2D, scalar_O3,
                  _mlir_ciface_transpose_2d_scalar_O3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);

/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  dataInput2D = transpose::allocArray<float>(_SIZE_ROW, _SIZE_COL);
  inputMemRef2D = MemRef<float, 2>(dataInput2D, sizesInput2D);

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Attain scalar output results as expected output results in verification.
  MemRef<float, 2> outputMemrefScalar(sizesOutput2D, 0);
  _mlir_ciface_transpose_2d_scalar_O0(&outputMemrefScalar, &inputMemRef2D);
  float *outputExpected = outputMemrefScalar.getData();

  MLIRVerification(outputExpected, _mlir_ciface_transpose_2d_scalar_O3,
                   "scalar_O3");
  /// [Step 3] Add your new method for verification.

  delete[] dataInput2D;
  return 0;
}
