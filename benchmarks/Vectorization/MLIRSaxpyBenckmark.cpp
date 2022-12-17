//===- MLIRSaxpyBenchmark.cpp --------------------------------------------===//
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
#include <buddy/core/Container.h>
#include <iostream>

// Declare the saxpy C interface.
extern "C" {
void _mlir_ciface_mlir_saxpy(MemRef<float, 1> *input1,
                              MemRef<float, 1> *input2,
                              MemRef<float, 1> *output);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRSaxpy1[1] = {10};
intptr_t sizesInputArrayMLIRSaxpy2[1] = {10};
intptr_t sizesOutputArrayMLIRSaxpy[1] = {10};
// Define the MemRef container for input1, input2, and output.
MemRef<float, 1> inputMLIRSaxpy1(sizesInputArrayMLIRSaxpy1, 2.4f);
MemRef<float, 1> inputMLIRSaxpy2(sizesInputArrayMLIRSaxpy2, 3.2f);
MemRef<float, 1> outputMLIRSaxpy(sizesOutputArrayMLIRSaxpy, 0.f);

static void MLIR_Saxpy(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_saxpy(&inputMLIRSaxpy1, &inputMLIRSaxpy2,
                               &outputMLIRSaxpy);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_Saxpy)->Arg(1);

// Generate result image.
void generateResultMLIRSaxpy() {
  // Define the MemRef descriptor for input1, intput2, and output.
  MemRef<float, 1> input1(sizesInputArrayMLIRSaxpy1, 2.4f);
  MemRef<float, 1> input2(sizesInputArrayMLIRSaxpy2, 3.2f);
  MemRef<float, 1> output(sizesOutputArrayMLIRSaxpy, 0.f);
  // Run the 1D Saxpy.
  _mlir_ciface_mlir_saxpy(&input1, &input2, &output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_Saxpy: MLIR Saxpy Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
