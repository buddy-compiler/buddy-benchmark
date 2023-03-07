//===- MLIRGccLoopsEx11Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex11 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex11(MemRef<int, 1> *A, MemRef<int, 1> *B,
                                    MemRef<int, 1> *C, MemRef<int, 1> *D);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx11[1] = {11};
intptr_t sizesOutputArrayMLIRGccLoopsEx11[1] = {5};
// Define the MemRef container for inputs and outputs.
int input_data_ex11[11] = {1,2,3,4,5,6,7,8,9,10,11};
MemRef<int, 1> inputMLIRGccLoopsEx11_1(input_data_ex11, sizesInputArrayMLIRGccLoopsEx11);
MemRef<int, 1> inputMLIRGccLoopsEx11_2(input_data_ex11, sizesInputArrayMLIRGccLoopsEx11);
MemRef<int, 1> outputMLIRGccLoopsEx11_1(sizesOutputArrayMLIRGccLoopsEx11, 0);
MemRef<int, 1> outputMLIRGccLoopsEx11_2(sizesOutputArrayMLIRGccLoopsEx11, 1);

static void MLIR_GccLoopsEx11(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex11(&outputMLIRGccLoopsEx11_1, &inputMLIRGccLoopsEx11_1,
                                     &inputMLIRGccLoopsEx11_2, &outputMLIRGccLoopsEx11_2);
    }
  }
} 

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx11)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx11() {
  // Define the MemRef descriptor for inputs and outputs.
  MemRef<int, 1> input_1(input_data_ex11, sizesInputArrayMLIRGccLoopsEx11);
  MemRef<int, 1> input_2(input_data_ex11, sizesInputArrayMLIRGccLoopsEx11);
  MemRef<int, 1> output_1(sizesOutputArrayMLIRGccLoopsEx11, 0);
  MemRef<int, 1> output_2(sizesOutputArrayMLIRGccLoopsEx11, 1);
  // Run the gccloopsex11.
  _mlir_ciface_mlir_gccloopsex11(&output_1, &input_1,
                                     &input_2, &output_2);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx11: MLIR GccLoopsEx11 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output_1.getSize(); i++) {
    std::cout << output_1.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output_2.getSize(); i++) {
    std::cout << output_2.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
