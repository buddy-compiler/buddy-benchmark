//===- MLIRGccLoopsEx13Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex13 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex13(MemRef<int, 2> *intput_1, MemRef<int, 2> *intput_2,
                                    MemRef<int, 1> *output);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx13[2] = {5,16};
intptr_t sizesOutputArrayMLIRGccLoopsEx13[1] = {5};
// Define the MemRef container for inputs and output.
MemRef<int, 2> inputMLIRGccLoopsEx13_1(sizesInputArrayMLIRGccLoopsEx13, 1);
MemRef<int, 2> inputMLIRGccLoopsEx13_2(sizesInputArrayMLIRGccLoopsEx13, 2);
MemRef<int, 1> outputMLIRGccLoopsEx13(sizesOutputArrayMLIRGccLoopsEx13, 0);

static void MLIR_GccLoopsEx13(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex13(&inputMLIRGccLoopsEx13_1, &inputMLIRGccLoopsEx13_2, &outputMLIRGccLoopsEx13);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx13)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx13() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<int, 2> input_1(sizesInputArrayMLIRGccLoopsEx13, 1);
  MemRef<int, 2> input_2(sizesInputArrayMLIRGccLoopsEx13, 2);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx13, 0);
  // Run the gccloopsex13.
  _mlir_ciface_mlir_gccloopsex13(&input_1, &input_2, &output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx13: MLIR GccLoopsEx13 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
