//===- BuddyFir.cpp
//---------------------------------------------------------===//
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
// This file implements the correctness checking for Buddy Fir function.
//
//===----------------------------------------------------------------------===//
#include "Utils/Container.h"
#include <stdio.h>

extern "C" {
void _mlir_ciface_conv1d_linalg(MemRef<float, 1> *inputBuddyConv1D,
                                MemRef<float, 1> *kernelBuddyConv1D,
                                MemRef<float, 1> *outputBuddyConv1D);

float *fir(float *input, float *kernel, float *output, int inputSize,
           int kernelSize, int outputSize) {
  MemRef<float, 1> *in =
      new MemRef<float, 1>(input, reinterpret_cast<intptr_t *>(&inputSize));
  MemRef<float, 1> *ker =
      new MemRef<float, 1>(kernel, reinterpret_cast<intptr_t *>(&kernelSize));
  MemRef<float, 1> *out =
      new MemRef<float, 1>(output, reinterpret_cast<intptr_t *>(&outputSize));
  _mlir_ciface_conv1d_linalg(in, ker, out);
  return out->getData();
}
}
