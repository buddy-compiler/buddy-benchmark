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

#include "gemmini.h"
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

intptr_t sizesA[2] = {_SIZE_M, _SIZE_K};
intptr_t sizesB[2] = {_SIZE_K, _SIZE_N};
intptr_t sizesC[2] = {_SIZE_M, _SIZE_N};
intptr_t sizesD[2] = {_SIZE_M, _SIZE_N};
int8_t *inputA = nullptr;
int8_t *inputB = nullptr;
int32_t *inputD = nullptr;
MemRef<int8_t, 2> inputAMemRef(sizesA);
MemRef<int8_t, 2> inputBMemRef(sizesB);
MemRef<int32_t, 2> inputDMemRef(sizesD);

// By inserting assembly code to obtain clock cycles
static uint64_t readCycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

// Gemmini native matmul function
void gemminiMatmul(
  int8_t *inputA, int8_t *inputB, 
  int8_t *outputC, int32_t *inputD){

  uint64_t start = readCycles();
  tiled_matmul_auto(
    _SIZE_M, _SIZE_N, _SIZE_K,
    inputA, inputB, inputD, outputC,
    _SIZE_K, _SIZE_N, _SIZE_N, _SIZE_N,
    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
    false,
    false, false,
    false, false,
    0,
    WS
  );
  uint64_t end = readCycles();
  std::cout << "Gemmini native matmul cycles: " << end - start << std::endl;

}

using MLIRFunctionType = void (*)(MemRef<int8_t, 2> *,
                                  MemRef<int8_t, 2> *, 
                                  MemRef<int8_t, 2> *,
                                  MemRef<int32_t, 2> *);
//  Verifies the result of an MLIR-based function against expected output.
void MLIRVerification(int8_t *outputExpected, MLIRFunctionType MLIRFunc,
                      const std::string &name) {
  MemRef<int8_t, 2> outputMemRef(sizesC, 0);
  uint64_t start = readCycles();
  MLIRFunc(&inputAMemRef, &inputBMemRef, &outputMemRef, &inputDMemRef);
  uint64_t end = readCycles();
  std::cout << name << " cycles: " << end - start << std::endl;
  int8_t *outputOptimized = outputMemRef.getData();
  matmul::verify<int8_t>(outputExpected, outputOptimized, _SIZE_M, _SIZE_N,
                        name);
}

// -----------------------------------------------------------------------------
// MLIR Benchmark. You can compare your new method with other methods here.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_gemmini_matmul1(MemRef<int8_t, 2>  *input0,
                                  MemRef<int8_t, 2>  *input1,
                                  MemRef<int8_t, 2>  *output,
                                  MemRef<int32_t, 2> *bias);
/// [Step 1] Add function of your new method.
}

// -----------------------------------------------------------------------------
// Main Function. You can verify the correctness of your new method here.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize input data.
  inputA = matmul::allocArray<int8_t>(_SIZE_M, _SIZE_K);
  inputB = matmul::allocArray<int8_t>(_SIZE_K, _SIZE_N);
  inputD = matmul::allocArray<int32_t>(_SIZE_M, _SIZE_N);
  inputAMemRef = MemRef<int8_t, 2>(inputA, sizesA);
  inputBMemRef = MemRef<int8_t, 2>(inputB, sizesB);
  inputDMemRef = MemRef<int32_t, 2>(inputD, sizesD);

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Attain Gemmini native matmul output results as expected output results in verification.
  int8_t* outputExpected = new int8_t[_SIZE_M * _SIZE_N];
  gemminiMatmul(inputA, inputB, outputExpected, inputD);

  /// [Step 3] Add your new method for verification.
  MLIRVerification(outputExpected, _mlir_ciface_gemmini_matmul1, "Buddy Gemmini MatMul");

  delete[] inputA;
  delete[] inputB;
  return 0;
}
