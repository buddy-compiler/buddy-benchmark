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
// This is the main file of Gemmini MatMul operation benchmark.
//
//===----------------------------------------------------------------------===//

#include "gemmini.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "Gemmini/Utils.h"
#include <buddy/Core/Container.h>

using namespace buddy::benchmark;

// -----------------------------------------------------------------------------
// Benchmark Configuration. You can change the number here as needed.
// -----------------------------------------------------------------------------

#define _NUM_ITER 1
#define _SIZE_M 16384
#define _SIZE_N 32
#define _SIZE_K 64
#define _BIAS 0
static float c_scale[1] = {1.0f};

// -----------------------------------------------------------------------------
// Include Kernel Functions.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_gemmini_matmul1(MemRef<int8_t, 2> *input0,
                                  MemRef<int8_t, 2> *input1,
                                  MemRef<int8_t, 2> *output,
                                  MemRef<int32_t, 2> *inputBias);
void _exo_matmul_4(const float *scale, bool act, const int8_t *A,
                   const int8_t *B, int8_t *C);
/// [Step 1] Add function of your new method.
}

// -----------------------------------------------------------------------------
// Global Variables.
// -----------------------------------------------------------------------------

static int8_t inputA[_SIZE_M * _SIZE_K] row_align(1);
static int8_t inputB[_SIZE_K * _SIZE_N] row_align(1);
static int32_t inputBias[_SIZE_M * _SIZE_N] row_align(1);
intptr_t sizesA[2] = {_SIZE_M, _SIZE_K};
intptr_t sizesB[2] = {_SIZE_K, _SIZE_N};
intptr_t sizesOutput[2] = {_SIZE_M, _SIZE_N};
intptr_t sizesBias[2] = {_SIZE_M, _SIZE_N};
MemRef<int8_t, 2> inputAMemRef(sizesA);
MemRef<int8_t, 2> inputBMemRef(sizesB);

// -----------------------------------------------------------------------------
// Benchmark Functions. The kernel functions are called here.
// -----------------------------------------------------------------------------

// Gemmini native matmul function.
// This function is used to get the expected output results for verification.
void nativeMatmul(int8_t *inputA, int8_t *inputB, int8_t *outputC,
                  int32_t *inputBias) {

  uint64_t start = gemmini::readCycles();
  tiled_matmul_auto(_SIZE_M, _SIZE_N, _SIZE_K, inputA, inputB, inputBias,
                    outputC, _SIZE_K, _SIZE_N, _SIZE_N, _SIZE_N,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
                    false, false, false, false, false, 0, WS);
  uint64_t end = gemmini::readCycles();
  std::cout << "Gemmini native matmul cycles: " << end - start << std::endl;
  // gemmini::printArrayInt8(outputC, _SIZE_M, _SIZE_N);
}

// Buddy Gemmini dialect matmul benchmark function.
// Verifies the result against expected output.
using MLIRFunctionType = void (*)(MemRef<int8_t, 2> *, MemRef<int8_t, 2> *,
                                  MemRef<int8_t, 2> *, MemRef<int32_t, 2> *);
void buddyMatmul(int8_t *outputExpected, MLIRFunctionType MLIRFunc,
                 const std::string &name) {
  int8_t output[_SIZE_M * _SIZE_N] row_align(1) = {0} ;
  MemRef<int8_t, 2> outputMemRef(sizesOutput, 0);
  outputMemRef = MemRef<int8_t, 2>(output, sizesOutput);
  MemRef<int32_t, 2> inputBiasMemRef(sizesBias, _BIAS);
  inputBiasMemRef = MemRef<int32_t, 2>(inputBias, sizesBias);
  uint64_t start = gemmini::readCycles();
  MLIRFunc(&inputAMemRef, &inputBMemRef, &outputMemRef, &inputBiasMemRef);
  uint64_t end = gemmini::readCycles();
  std::cout << name << " cycles: " << end - start << std::endl;
  int8_t *outputOptimized = outputMemRef.getData();
  gemmini::verify<int8_t>(outputExpected, outputOptimized, _SIZE_M, _SIZE_N,
                          name);
  // gemmini::printArrayInt8(outputOptimized, _SIZE_M, _SIZE_N);
}

// Exo-lang matmul benchmark function.
void exoMatmul(int8_t *outputExpected) {
  static int8_t outputExoMatmul[_SIZE_M * _SIZE_N] row_align(1);
  for (int i = 0; i < _SIZE_M * _SIZE_N; i++) {
    outputExoMatmul[i] = 0;
  }
  uint64_t start = gemmini::readCycles();
  _exo_matmul_4(c_scale, false, inputA, inputB, outputExoMatmul);
  uint64_t end = gemmini::readCycles();
  std::cout << "Exo-lang Gemmini MatMul cycles: " << end - start << std::endl;
  gemmini::verify<int8_t>(outputExpected, outputExoMatmul, _SIZE_M, _SIZE_N,
                          "Exo-lang Gemmini MatMul");
  // gemmini::printArrayInt8(outputExoMatmul, _SIZE_M, _SIZE_N);
}

// -----------------------------------------------------------------------------
// Main Function.
// -----------------------------------------------------------------------------

int main() {
  // Initialize input data.
  for (int i = 0; i < _SIZE_M; i++) {
    for (int j = 0; j < _SIZE_K; j++) {
      inputA[(_SIZE_K)*i + j] = 1;
      // inputA[(_SIZE_K)*i + j] = i + j * 2;
    }
  }

  for (int i = 0; i < _SIZE_K; i++) {
    for (int j = 0; j < _SIZE_N; j++) {
      inputB[(_SIZE_N)*i + j] = 1;
      // inputB[(_SIZE_N)*i + j] = j * 3 + i;
    }
  }

  for (int i = 0; i < _SIZE_M; i++) {
    for (int j = 0; j < _SIZE_N; j++) {
      inputBias[(_SIZE_N)*i + j] = _BIAS;
    }
  }

  inputAMemRef = MemRef<int8_t, 2>(inputA, sizesA);
  inputBMemRef = MemRef<int8_t, 2>(inputB, sizesB);

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;

  int8_t outputExpected[_SIZE_M * _SIZE_N] row_align(1) = {0} ;

  nativeMatmul(inputA, inputB, outputExpected, inputBias);

  buddyMatmul(outputExpected, _mlir_ciface_gemmini_matmul1,
              "Buddy Gemmini MatMul");

  exoMatmul(outputExpected);

  return 0;
}
