//===- MLIRGccLoopsEx10bBenchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex10b C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex10b(MemRef<short, 1> *sb, MemRef<int, 1> *ia);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx10b[1] = {10};
intptr_t sizesOutputArrayMLIRGccLoopsEx10b[1] = {10};
// Define the MemRef container for input and output.
short input_data_ex10b[10] = {1,2,3,4,5,6,7,8,9,10};
MemRef<short, 1> inputMLIRGccLoopsEx10b(input_data_ex10b, sizesInputArrayMLIRGccLoopsEx10b);
MemRef<int, 1> outputMLIRGccLoopsEx10b(sizesOutputArrayMLIRGccLoopsEx10b, 0);

static void MLIR_GccLoopsEx10b(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex10b(&inputMLIRGccLoopsEx10b, &outputMLIRGccLoopsEx10b);
    }
  }
} 

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx10b)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx10b() {
  // Define the MemRef descriptor for input and output.
  short input_data_ex10b[10] = {1,2,3,4,5,6,7,8,9,10};
  MemRef<short, 1> input(input_data_ex10b, sizesInputArrayMLIRGccLoopsEx10b);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx10b, 0);
  // Run the gccloopsex10b.
  _mlir_ciface_mlir_gccloopsex10b(&input, &output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx10b: MLIR GccLoopsEx10b Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
