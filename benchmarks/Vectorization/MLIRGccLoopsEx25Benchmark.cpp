//===- MLIRGccLoopsEx25Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex25 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex25(MemRef<int, 1> *dj, MemRef<float, 1> *da,  MemRef<float, 1> *db,
                                    MemRef<float, 1> *dc,  MemRef<float, 1> *dd);
}

// Define input and output sizes.
intptr_t sizesArrayMLIRGccLoopsEx25[1] = {10};
// Define the MemRef container for inputs and output.
float input_data_ex25_1[10] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.1,10.2};
float input_data_ex25_2[10] = {1.1,2.4,3.3,4.6,5.5,6.8,7.7,8.8,9.0,10.3};
float input_data_ex25_3[10] = {1.1,2.4,3.3,4.6,5.5,6.8,7.7,8.8,9.0,10.3};
float input_data_ex25_4[10] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.1,10.2};
MemRef<float, 1> inputMLIRGccLoopsEx25_1(input_data_ex25_1, sizesArrayMLIRGccLoopsEx25);
MemRef<float, 1> inputMLIRGccLoopsEx25_2(input_data_ex25_2, sizesArrayMLIRGccLoopsEx25);
MemRef<float, 1> inputMLIRGccLoopsEx25_3(input_data_ex25_3, sizesArrayMLIRGccLoopsEx25);
MemRef<float, 1> inputMLIRGccLoopsEx25_4(input_data_ex25_4, sizesArrayMLIRGccLoopsEx25);
MemRef<int, 1> outputMLIRGccLoopsEx25(sizesArrayMLIRGccLoopsEx25, 0);

static void MLIR_GccLoopsEx25(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex25(&outputMLIRGccLoopsEx25, &inputMLIRGccLoopsEx25_1, &inputMLIRGccLoopsEx25_2,
                                     &inputMLIRGccLoopsEx25_3, &inputMLIRGccLoopsEx25_4);
    }
  }
} 

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx25)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx25() {
  // Define the MemRef descriptor for inputs and output.
  float input_data_ex25_1[10] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.1,10.2};
  float input_data_ex25_2[10] = {1.1,2.4,3.3,4.6,5.5,6.8,7.7,8.8,9.0,10.3};
  float input_data_ex25_3[10] = {1.1,2.4,3.3,4.6,5.5,6.8,7.7,8.8,9.0,10.3};
  float input_data_ex25_4[10] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.1,10.2};
  MemRef<float, 1> input_1(input_data_ex25_1, sizesArrayMLIRGccLoopsEx25);
  MemRef<float, 1> input_2(input_data_ex25_2, sizesArrayMLIRGccLoopsEx25);
  MemRef<float, 1> input_3(input_data_ex25_3, sizesArrayMLIRGccLoopsEx25);
  MemRef<float, 1> input_4(input_data_ex25_4, sizesArrayMLIRGccLoopsEx25);
  MemRef<int, 1> output(sizesArrayMLIRGccLoopsEx25, 0);
  // Run the gccloopsex25.
  _mlir_ciface_mlir_gccloopsex25(&output, &input_1, &input_2,
                                     &input_3, &input_4);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx25: MLIR GccLoopsEx25 Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
