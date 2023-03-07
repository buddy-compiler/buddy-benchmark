//===- MLIRGccLoopsEx21Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex21 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex21(MemRef<int, 1> *b);
}

// Define input and output sizes.
intptr_t sizesArrayMLIRGccLoopsEx21[1] = {10};
// Define the MemRef container.
int input_data_ex21[10] = {1,2,3,4,5,6,7,8,9,10};
MemRef<int, 1> MLIRGccLoopsEx21(input_data_ex21, sizesArrayMLIRGccLoopsEx21);

static void MLIR_GccLoopsEx21(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex21(&MLIRGccLoopsEx21);
    }
  }
} 

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx21)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx21() {
  // Define the MemRef descriptor.
  int input_data_ex21[10] = {1,2,3,4,5,6,7,8,9,10};
  MemRef<int, 1> MLIRGccLoopsEx21(input_data_ex21, sizesArrayMLIRGccLoopsEx21);
  // Run the gccloopsex21.
  _mlir_ciface_mlir_gccloopsex21(&MLIRGccLoopsEx21);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx21: MLIR GccLoopsEx21 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < MLIRGccLoopsEx21.getSize(); i++) {
    std::cout << MLIRGccLoopsEx21.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
