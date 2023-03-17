//===- MLIRGccLoopsEx8Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex8 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex8(int x, MemRef<int, 2> *output);
}

// Define input and output sizes.
intptr_t sizesOutputArrayMLIRGccLoopsEx8[2] = {5,2};
// Define the MemRef container for output.
int x_ex8 = 3;
MemRef<int, 2> outputMLIRGccLoopsEx8(sizesOutputArrayMLIRGccLoopsEx8, 0);

static void MLIR_GccLoopsEx8(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex8(x_ex8, &outputMLIRGccLoopsEx8);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx8)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx8() {
  // Define the MemRef descriptor for output.
  int x = 3;
  MemRef<int, 2> output(sizesOutputArrayMLIRGccLoopsEx8, 0);
  // Run the gccloopsex8.
  _mlir_ciface_mlir_gccloopsex8(x, &output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx8: MLIR GccLoopsEx8 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
