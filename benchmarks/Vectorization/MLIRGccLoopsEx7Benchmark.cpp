//===- MLIRGccLoopsEx7Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex7 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex7(int x, MemRef<int, 1> *output,
                              MemRef<int, 1> *input);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx7[1] = {13};
intptr_t sizesOutputArrayMLIRGccLoopsEx7[1] = {10};
// Define the MemRef container for input and output.
int input_data_ex7[13] = {1,2,3,4,5,6,7,8,9,10,11,12,13};
int x = 3;
MemRef<int, 1> inputMLIRGccLoopsEx7(input_data_ex7, sizesInputArrayMLIRGccLoopsEx7);
MemRef<int, 1> outputMLIRGccLoopsEx7(sizesOutputArrayMLIRGccLoopsEx7, 0);

static void MLIR_GccLoopsEx7(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex7(x, &outputMLIRGccLoopsEx7, &inputMLIRGccLoopsEx7);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx7)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx7() {
  // Define the MemRef descriptor for input and output.
  int x = 3;
  MemRef<int, 1> input(input_data_ex7, sizesInputArrayMLIRGccLoopsEx7);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx7, 0);
  // Run the gccloopsex7.
  _mlir_ciface_mlir_gccloopsex7(x, &output, &input);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx7: MLIR GccLoopsEx7 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
