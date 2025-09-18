//===- MLIRMatVecBenchmark.cpp --------------------------------------------===//
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

// Declare the matvec C interface.
extern "C" {
void _mlir_ciface_mlir_matvec(MemRef<float, 2> *input1,
                              MemRef<float, 2> *input2,
                              MemRef<float, 2> *output);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRMatVec1[2] = {5, 3};
intptr_t sizesInputArrayMLIRMatVec2[2] = {3, 2};
intptr_t sizesOutputArrayMLIRMatVec[2] = {5, 2};
// Define the MemRef container for input1, input2, and output.
MemRef<float, 2> inputMLIRMatVec1(sizesInputArrayMLIRMatVec1, 2);
MemRef<float, 2> inputMLIRMatVec2(sizesInputArrayMLIRMatVec2, 3);
MemRef<float, 2> outputMLIRMatVec(sizesOutputArrayMLIRMatVec, 0);

static void MLIR_MatVec(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_matvec(&inputMLIRMatVec1, &inputMLIRMatVec2,
                               &outputMLIRMatVec);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_MatVec)->Arg(1);

// Generate result image.
void generateResultMLIRMatVec() {
  // Define the MemRef descriptor for input1, intput2, and output.
  MemRef<float, 2> input1(sizesInputArrayMLIRMatVec1, 2);
  MemRef<float, 2> input2(sizesInputArrayMLIRMatVec2, 3);
  MemRef<float, 2> output(sizesOutputArrayMLIRMatVec, 0);
  // Run the 2D matvec.
  _mlir_ciface_mlir_matvec(&input1, &input2, &output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_MatVec: MLIR MatVec Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
