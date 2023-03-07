//===- MLIRGccLoopsEx2aBenchmark.cpp --------------------------------------------===//
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

// Declare the GccLoopsEx2a C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex2a(MemRef<int, 1> *output,
                              size_t input1,
                              int input2);
}

// Define input and output sizes.
intptr_t sizesOutputArrayMLIRGccLoopsEx2a[1] = {10};
// Define the MemRef container for inputs and output.
size_t input1 = 10;
int input2 = 16;
MemRef<int, 1> outputMLIRGccLoopsEx2a(sizesOutputArrayMLIRGccLoopsEx2a, 0);

static void MLIR_GccLoopsEx2a(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex2a(&outputMLIRGccLoopsEx2a, input1,
                               input2);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx2a)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx2a() {
  // Define the MemRef descriptor for inputs and output.
  size_t input1 = 10;
  int input2 = 16;
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx2a, 0);
  // Run the GccLoopsEx2a.
  _mlir_ciface_mlir_gccloopsex2a(&output, input1, input2);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx2a: MLIR GccLoopsEx2a Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
