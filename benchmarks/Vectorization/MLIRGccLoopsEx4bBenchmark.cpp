//===- MLIRGccLoopsEx4bBenchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex4b C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex4b(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx4b_1[1] = {11};
intptr_t sizesInputArrayMLIRGccLoopsEx4b_2[1] = {13};
intptr_t sizesOutputArrayMLIRGccLoopsEx4b[1] = {10};
// Define the MemRef container for inputs and output.
int input_1[11] = {1,2,3,4,5,6,7,8,9,10,11};
int input_2[13] = {1,2,3,4,5,6,7,8,9,10,11,12,13};
MemRef<int, 1> inputMLIRGccLoopsEx4b_1(input_1, sizesInputArrayMLIRGccLoopsEx4b_1);
MemRef<int, 1> inputMLIRGccLoopsEx4b_2(input_2, sizesInputArrayMLIRGccLoopsEx4b_2);
MemRef<int, 1> outputMLIRGccLoopsEx4b(sizesOutputArrayMLIRGccLoopsEx4b, 0);

static void MLIR_GccLoopsEx4b(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex4b(&outputMLIRGccLoopsEx4b, &inputMLIRGccLoopsEx4b_1,
                               &inputMLIRGccLoopsEx4b_2);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx4b)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx4b() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<int, 1> input1(input_1, sizesInputArrayMLIRGccLoopsEx4b_1);
  MemRef<int, 1> input2(input_2, sizesInputArrayMLIRGccLoopsEx4b_2);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx4b, 0);
  // Run the gccloopsex4b.
  _mlir_ciface_mlir_gccloopsex4b(&output, &input1, &input2);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx4b: MLIR GccLoopsEx4b Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
