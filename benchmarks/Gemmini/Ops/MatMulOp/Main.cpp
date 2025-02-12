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
#include <buddy/Core/Container.h>

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the number here as needed.
// -----------------------------------------------------------------------------

#define _NUM_ITER 1
#define _SIZE_M 32
#define _SIZE_N 32
#define _SIZE_K 32

// -----------------------------------------------------------------------------
// Global Variables and Functions. No need to change the code here.
// -----------------------------------------------------------------------------

static uint64_t readCycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

intptr_t sizesA[2] = {_SIZE_M, _SIZE_K};
intptr_t sizesB[2] = {_SIZE_K, _SIZE_N};
intptr_t sizesC[2] = {_SIZE_M, _SIZE_N};
intptr_t sizesD[2] = {_SIZE_M, _SIZE_N};
int8_t *inputA = nullptr;
int8_t *inputB = nullptr;
MemRef<int8_t, 2> inputAMemRef(sizesA);
MemRef<int8_t, 2> inputBMemRef(sizesB);

// Runs the provided MatMul function for benchmarking.
// template <typename Func>
// void DL_OPS_MATMUL(benchmark::State &state, Func func) {
//   MemRef<float, 2> outputMemRef(sizesC, 0.0);
//   for (auto _ : state) {
//     func(&inputAMemRef, &inputBMemRef, &outputMemRef);
//   }
//   benchmark::DoNotOptimize(outputMemRef);
// }

// using MLIRFunctionType = void (*)(MemRef<int8_t, 2> *,
//                                   MemRef<int8_t, 2> *, 
//                                   MemRef<int8_t, 2> *,
//                                   MemRef<int32_t, 2> *);
// //  Verifies the result of an MLIR-based function against expected output.
// void MLIRVerification(float *outputExpected, MLIRFunctionType MLIRFunc,
//                       const std::string &name) {
//   MemRef<float, 2> outputMemRef(sizesC, 0);
//   MLIRFunc(&inputAMemRef, &inputBMemRef, &outputMemRef);
//   float *outputOptimized = outputMemRef.getData();
//   matmul::verify<float>(outputExpected, outputOptimized, _SIZE_M, _SIZE_N,
//                         name);
// }

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_gemmini_matmul1(MemRef<int8_t, 2>  *input0,
                                  MemRef<int8_t, 2>  *input1,
                                  MemRef<int8_t, 2>  *bias,
                                  MemRef<int32_t, 2> *output);
/// [Step 1] Add function of your new method.
}
// BENCHMARK_CAPTURE(DL_OPS_MATMUL, scalar_O0, _mlir_ciface_gemmini_matmul1)
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(_NUM_ITER);
/// [Step 2] Call GoogleBenchmark function to run your new method.

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  inputA = matmul::allocArray<int8_t>(_SIZE_M, _SIZE_K);
  inputB = matmul::allocArray<int8_t>(_SIZE_K, _SIZE_N);
  inputAMemRef = MemRef<int8_t, 2>(inputA, sizesA);
  inputBMemRef = MemRef<int8_t, 2>(inputB, sizesB);
  MemRef<int8_t, 2> inputDMemRef(sizesD, 0);

  // for (int i = 0; i < _SIZE_M; i++) {
  //   for (int j = 0; j < _SIZE_N; j++) {
  //     std::cout << static_cast<int>(inputA[i * _SIZE_N + j]) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // Run benchmark.
  // ::benchmark::Initialize(&argc, argv);
  // ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Attain scalar output results as expected output results in verification.
  MemRef<int32_t, 2> outputMemrefScalar(sizesC, 0);
  _mlir_ciface_gemmini_matmul1(&inputAMemRef, &inputBMemRef, &inputDMemRef, &outputMemrefScalar);
  int32_t *outputExpected = outputMemrefScalar.getData();

  /// [Step 3] Add your new method for verification.

  delete[] inputA;
  delete[] inputB;
  return 0;
}
