//===- MLIRGccLoopsEx14Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex14 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex14(MemRef<int, 2> *intput_1, MemRef<int, 2> *intput_2,
                                    MemRef<int, 1> *output);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx14_1[2] = {6,3};
intptr_t sizesInputArrayMLIRGccLoopsEx14_2[2] = {4,3};
intptr_t sizesOutputArrayMLIRGccLoopsEx14[1] = {2};
// Define the MemRef container for inputs and output.
MemRef<int, 2> inputMLIRGccLoopsEx14_1(sizesInputArrayMLIRGccLoopsEx14_1, 3);
MemRef<int, 2> inputMLIRGccLoopsEx14_2(sizesInputArrayMLIRGccLoopsEx14_2, 2);
MemRef<int, 1> outputMLIRGccLoopsEx14(sizesOutputArrayMLIRGccLoopsEx14, 0);

static void MLIR_GccLoopsEx14(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex14(&inputMLIRGccLoopsEx14_1, &inputMLIRGccLoopsEx14_2, &outputMLIRGccLoopsEx14);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx14)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx14() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<int, 2> input_1(sizesInputArrayMLIRGccLoopsEx14_1, 3);
  MemRef<int, 2> input_2(sizesInputArrayMLIRGccLoopsEx14_2, 2);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx14, 0);
  // Run the gccloopsex14.
  _mlir_ciface_mlir_gccloopsex14(&input_1, &input_2, &output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx14: MLIR GccLoopsEx14 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
