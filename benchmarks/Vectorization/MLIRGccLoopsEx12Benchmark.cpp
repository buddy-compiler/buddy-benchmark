//===- MLIRGccLoopsEx12Benchmark.cpp --------------------------------------------===//
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

// Declare the GccLoopsEx12 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex12(MemRef<int, 1> *output);
}

// Define input and output sizes.
intptr_t sizesOutputArrayMLIRGccLoopsEx12[1] = {10};
// Define the MemRef container for output.
MemRef<int, 1> outputMLIRGccLoopsEx12(sizesOutputArrayMLIRGccLoopsEx12, 0);

static void MLIR_GccLoopsEx12(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex12(&outputMLIRGccLoopsEx12);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx12)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx12() {
  // Define the MemRef descriptor for output.
  MemRef<int, 1> output(sizesOutputArrayMLIRGccLoopsEx12, 0);
  // Run the GccLoopsEx12.
  _mlir_ciface_mlir_gccloopsex12(&output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx12: MLIR GccLoopsEx12 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
