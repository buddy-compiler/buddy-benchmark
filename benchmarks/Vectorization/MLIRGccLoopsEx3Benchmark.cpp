//===- MLIRGccLoopsEx3Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex3 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex3(size_t n,
                              MemRef<int, 1> *output,
                              MemRef<int, 1> *input);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx3[1] = {12};
intptr_t sizesOutputArrayMLIRGccLoopsEx3[1] = {12};
// Define the MemRef container for n, intput, and output.
size_t n = 12;
MemRef<int, 1> inputMLIRGccLoopsEx3(sizesInputArrayMLIRGccLoopsEx3, 2);
MemRef<int, 1> outputMLIRGccLoopsEx3(sizesOutputArrayMLIRGccLoopsEx3, 0);

static void MLIR_GccLoopsEx3(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex3(n, &outputMLIRGccLoopsEx3,
                               &inputMLIRGccLoopsEx3);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx3)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx3() {
  // Define the MemRef descriptor for n, intput, and output.
  size_t n = 12;
  MemRef<int, 1> input(sizesInputArrayMLIRGccLoopsEx3, 2);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx3, 0);
  // Run the gccloopsex3.
  _mlir_ciface_mlir_gccloopsex3(n, &output, &input);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx3: MLIR GccLoopsEx3 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
