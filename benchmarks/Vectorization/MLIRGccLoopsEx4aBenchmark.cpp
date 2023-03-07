//===- MLIRGccLoopsEx4aBenchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex4a C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex4a(size_t n,
                              MemRef<int, 1> *output,
                              MemRef<int, 1> *input);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx4a[1] = {12};
intptr_t sizesOutputArrayMLIRGccLoopsEx4a[1] = {12};
// Define the MemRef container for intput, and output.
size_t n_4a = 12;
MemRef<int, 1> inputMLIRGccLoopsEx4a(sizesInputArrayMLIRGccLoopsEx4a, 2);
MemRef<int, 1> outputMLIRGccLoopsEx4a(sizesOutputArrayMLIRGccLoopsEx4a, 0);

static void MLIR_GccLoopsEx4a(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex4a(n_4a, &outputMLIRGccLoopsEx4a,
                               &inputMLIRGccLoopsEx4a);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx4a)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx4a() {
  // Define the MemRef descriptor for intput, and output.
  size_t n = 12;
  MemRef<int, 1> input(sizesInputArrayMLIRGccLoopsEx4a, 2);
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx4a, 0);
  // Run the gccloopsex4a.
  _mlir_ciface_mlir_gccloopsex4a(n, &output, &input);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx4a: MLIR GccLoopsEx4a Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
