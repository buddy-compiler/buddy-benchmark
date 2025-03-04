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
// This is the main file of Gemmini Conv operation benchmark.
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

#define _BATCH_SIZE 4
#define _IN_CH 64
#define _OUT_CH 64
#define _IN_DIM 58
#define _OUT_DIM 56
#define _KERNEL_DIM 3
static float c_scale[1] = {1.0f};
static int32_t _BIAS = 1;


// -----------------------------------------------------------------------------
// Include Kernel Functions.
// -----------------------------------------------------------------------------

extern "C" {
void _mlir_ciface_gemmini_conv_3(MemRef<int8_t, 2> *input0,
                                 MemRef<int8_t, 2> *input1,
                                 MemRef<int32_t, 2> *inputBias,
                                 MemRef<int8_t, 2> *output);
// void _exo_conv_3(int8_t* output, const int32_t* bias, 
//                  const int8_t* inp, const int8_t* weights, 
//                  bool act, const float* scale );
}

// -----------------------------------------------------------------------------
// Global Variables.
// -----------------------------------------------------------------------------

static int8_t input[_BATCH_SIZE * _IN_DIM * _IN_DIM * _IN_CH] row_align(1); // NHWC 
static int8_t weights[_KERNEL_DIM * _KERNEL_DIM * _IN_CH * _OUT_CH] row_align(1); // FHWC
static int32_t inputBias[_IN_CH] row_align(1);
static int8_t output[_BATCH_SIZE * _OUT_DIM * _OUT_DIM * _OUT_CH] row_align(1); // NHW C

intptr_t inputSizes[4] = {_BATCH_SIZE, _IN_DIM, _IN_DIM, _IN_CH};
intptr_t weightsSizes[4] = {_KERNEL_DIM * _KERNEL_DIM, _IN_CH * _OUT_CH};
intptr_t biasSizes[1] = {_OUT_CH};
intptr_t outputSizes[2] = {_BATCH_SIZE * _OUT_DIM * _OUT_DIM, _OUT_CH};

MemRef<int8_t, 2> inputAMemRef(inputSizes);
MemRef<int8_t, 2> inputBMemRef(weightsSizes);

// -----------------------------------------------------------------------------
// Benchmark Functions. The kernel functions are called here.
// -----------------------------------------------------------------------------

// Gemmini native convolution function.
// This function is used to get the expected output results for verification.
void nativeConv(int8_t *inputA, int8_t *inputB, int8_t *outputC,
                  int32_t *inputBias) {

  uint64_t start = gemmini::readCycles();
  tiled_conv_auto(_BATCH_SIZE, _IN_DIM, _IN_DIM, _IN_CH, _OUT_CH, 
                  _OUT_DIM, _OUT_DIM, 1, 1, 1, 0, _KERNEL_DIM,
                  false, false, false, false, false,
                  inputA, inputB, inputBias, outputC,
                  NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, WS);
  uint64_t end = gemmini::readCycles();
  
  std::cout << "Gemmini native Conv cycles: " << end - start << std::endl;
}

// Buddy Gemmini dialect Conv benchmark function.
// Verifies the result against expected output.
using MLIRFunctionType = void (*)(MemRef<int8_t, 2> *, MemRef<int8_t, 2> *,
                                  MemRef<int32_t, 2> *, MemRef<int8_t, 2> *);
void buddyConv(int8_t *outputExpected, MLIRFunctionType MLIRFunc,
               const std::string &name) {
  MemRef<int8_t, 2> outputMemRef(outputSizes, 0);
  MemRef<int32_t, 2> biasMemRef(biasSizes, _BIAS);

  uint64_t start = gemmini::readCycles();
  MLIRFunc(&inputAMemRef, &inputBMemRef, &biasMemRef, &outputMemRef);
  uint64_t end = gemmini::readCycles();
  
  std::cout << name << " cycles: " << end - start << std::endl;
  gemmini::verify<int8_t>(outputExpected, outputMemRef.getData(), 
                         _BATCH_SIZE*_OUT_DIM*_OUT_DIM, _OUT_CH, name);
}


// Exo-lang Conv benchmark function.
// void exoConv(int8_t *outputExpected) {
//   static int8_t outputExo[_BATCH_SIZE * _OUT_DIM * _OUT_DIM * _OUT_CH] = {0};
  
//   uint64_t start = gemmini::readCycles();
//   _exo_conv_3(outputExo, inputBias, input, weights, false, c_scale);
//   uint64_t end = gemmini::readCycles();
  
//   std::cout << "Exo-lang Gemmini Conv cycles: " << end - start << std::endl;
//   gemmini::verify<int8_t>(outputExpected, outputExo, 
//                          _BATCH_SIZE * _OUT_DIM * _OUT_DIM, _OUT_CH,
//                          "Exo-lang Gemmini Conv");
// }

// -----------------------------------------------------------------------------
// Main Function.
// -----------------------------------------------------------------------------

int main() {
  // Initialize input data.
  for (int b = 0; b < _BATCH_SIZE; ++b) {
      for (int h = 0; h < _IN_DIM; ++h) {
        for (int w = 0; w < _IN_DIM; ++w) {
          for (int c = 0; c < _IN_CH; ++c) {
            input[((b * _IN_DIM + h) * _IN_DIM + w) * _IN_CH + c] = 1;
          }
        }
      }
    }
  for (int kh = 0; kh <  _KERNEL_DIM; ++kh) {
    for (int kw = 0; kw <  _KERNEL_DIM; ++kw) {
      for (int ic = 0; ic < _IN_CH; ++ic) {
        for (int oc = 0; oc < _OUT_CH; ++oc) {
          weights[((kh* _KERNEL_DIM + kw)*_IN_CH + ic)*_OUT_CH + oc] = 1;
        }
      }
    }
  }

  for (int oc = 0; oc < _OUT_CH; ++oc) {
    inputBias[oc] = _BIAS;
  }

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;

  int8_t outputExpected[_BATCH_SIZE * _OUT_DIM * _OUT_DIM * _OUT_CH] row_align(1) = {0};
  nativeConv(input, weights, outputExpected, inputBias);
  buddyConv(outputExpected, _mlir_ciface_gemmini_conv_3, "Buddy Gemmini Conv");
  // exoConv(outputExpected);

  return 0;
}
