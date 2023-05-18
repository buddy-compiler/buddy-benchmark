//===- MLIRGccLoopsEx23Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex23 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex23(MemRef<short, 1> *src, MemRef<int, 1> *dst);
}

// Define input and output sizes.
intptr_t sizesArrayMLIRGccLoopsEx23[1] = {10};
// Define the MemRef container for input and output.
short input_data_ex23[10] = {1,2,3,4,5,6,7,8,9,10};
MemRef<short, 1> inputMLIRGccLoopsEx23(input_data_ex23, sizesArrayMLIRGccLoopsEx23);
MemRef<int, 1> outputMLIRGccLoopsEx23(sizesArrayMLIRGccLoopsEx23, 1);

static void MLIR_GccLoopsEx23(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex23(&inputMLIRGccLoopsEx23, &outputMLIRGccLoopsEx23);
    }
  }
} 

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx23)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx23() {
  // Define the MemRef descriptor for input and output.
  short input_data_ex23[10] = {1,2,3,4,5,6,7,8,9,10};
  MemRef<short, 1> input(input_data_ex23, sizesArrayMLIRGccLoopsEx23);
  MemRef<int, 1> output(sizesArrayMLIRGccLoopsEx23, 1);
  // Run the gccloopsex23.
  _mlir_ciface_mlir_gccloopsex23(&input, &output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx23: MLIR GccLoopsEx23 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
