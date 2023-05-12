//===- MLIRGccLoopsEx24Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex24 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex24(MemRef<short, 1> *ic, MemRef<float, 1> *fa,  MemRef<float, 1> *fb);
}

// Define input and output sizes.
intptr_t sizesArrayMLIRGccLoopsEx24[1] = {10};
// Define the MemRef container for inputs and output.
float input_data_ex24_1[10] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.1,10.2};
float input_data_ex24_2[10] = {1.1,2.4,3.3,4.6,5.5,6.8,7.7,8.8,9.0,10.3};
MemRef<float, 1> inputMLIRGccLoopsEx24_1(input_data_ex24_1, sizesArrayMLIRGccLoopsEx24);
MemRef<float, 1> inputMLIRGccLoopsEx24_2(input_data_ex24_2, sizesArrayMLIRGccLoopsEx24);
MemRef<short, 1> outputMLIRGccLoopsEx24(sizesArrayMLIRGccLoopsEx24, 0);

static void MLIR_GccLoopsEx24(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex24(&outputMLIRGccLoopsEx24, &inputMLIRGccLoopsEx24_1, &inputMLIRGccLoopsEx24_2);
    }
  }
} 

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx24)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx24() {
  // Define the MemRef descriptor for inputs and output.
  float input_data_ex24_1[10] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.1,10.2};
  float input_data_ex24_2[10] = {1.1,2.4,3.3,4.6,5.5,6.8,7.7,8.8,9.0,10.3};
  MemRef<float, 1> input_1(input_data_ex24_1, sizesArrayMLIRGccLoopsEx24);
  MemRef<float, 1> input_2(input_data_ex24_2, sizesArrayMLIRGccLoopsEx24);
  MemRef<short, 1> output(sizesArrayMLIRGccLoopsEx24, 0);
  // Run the gccloopsex24.
  _mlir_ciface_mlir_gccloopsex24(&output, &input_1, &input_2);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx24: MLIR GccLoopsEx24 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
