//===- MLIRGccLoopsEx2bBenchmark.cpp --------------------------------------------===//
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
// This file implements the benchmark for buddy-opt tool in buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>

// Declare the GccLoopsEx2b C interface.
extern "C" {
size_t _mlir_ciface_mlir_gccloopsex2b(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2,
                              size_t input3);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx2b1[1] = {10};
intptr_t sizesInputArrayMLIRGccLoopsEx2b2[1] = {10};
intptr_t sizesOutputArrayMLIRGccLoopsEx2b[1] = {10};
// Define the MemRef container for inputs and output.
MemRef<int, 1> inputMLIRGccLoopsEx2b1(sizesInputArrayMLIRGccLoopsEx2b1, 5);
MemRef<int, 1> inputMLIRGccLoopsEx2b2(sizesInputArrayMLIRGccLoopsEx2b2, 6);
size_t input3 = 10;
MemRef<int, 1> outputMLIRGccLoopsEx2b1(sizesOutputArrayMLIRGccLoopsEx2b, 0);
int outputMLIRGccLoopsEx2b2 = 0;

static void MLIR_GccLoopsEx2b(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
        outputMLIRGccLoopsEx2b2 = _mlir_ciface_mlir_gccloopsex2b(&outputMLIRGccLoopsEx2b1, &inputMLIRGccLoopsEx2b1,
                               &inputMLIRGccLoopsEx2b2, input3);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx2b)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx2b() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<int, 1> input1(sizesInputArrayMLIRGccLoopsEx2b1, 5);
  MemRef<int, 1> input2(sizesInputArrayMLIRGccLoopsEx2b2, 6);
  size_t input3 = 10;
  MemRef<int, 1> output1(sizesOutputArrayMLIRGccLoopsEx2b, 0);
  int output2 = 0;
  // Run the GccLoopsEx2b.
  output2 = _mlir_ciface_mlir_gccloopsex2b(&output1, &input1, &input2, input3);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx2b: MLIR GccLoopsEx2b Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output1.getSize(); i++) {
    std::cout << output1.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << output2 << std::endl;
}
