//===- MLIRGccLoopsEx9Benchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex9 C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex9(int* ret, MemRef<int, 1> *intput_1, 
                                                MemRef<int, 1> *intput_2);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx9[1] = {10};
intptr_t sizesOutputArrayMLIRGccLoopsEx9[1] = {1};
// Define the MemRef container for inputs and output.
int input1_data_ex9[10] = {1,2,3,4,5,6,7,8,9,10};
int input2_data_ex9[10] = {1,1,1,1,1,1,1,1,1,1};
int ret = 0;
int *ret_ex9 = &ret;
MemRef<int, 1> inputMLIRGccLoopsEx9_1(input1_data_ex9, sizesInputArrayMLIRGccLoopsEx9);
MemRef<int, 1> inputMLIRGccLoopsEx9_2(input2_data_ex9, sizesInputArrayMLIRGccLoopsEx9);

static void MLIR_GccLoopsEx9(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex9(ret_ex9, &inputMLIRGccLoopsEx9_1, &inputMLIRGccLoopsEx9_2);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx9)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx9() {
  // Define the MemRef descriptor for inputs and output.
  int input1_data_ex9[10] = {1,2,3,4,5,6,7,8,9,10};
  int input2_data_ex9[10] = {1,1,1,1,1,1,1,1,1,1};
  int ret = 0;
  int *ret_ex9 = &ret;
  MemRef<int, 1> input_1(input1_data_ex9, sizesInputArrayMLIRGccLoopsEx9);
  MemRef<int, 1> input_2(input2_data_ex9, sizesInputArrayMLIRGccLoopsEx9);
  // Run the gccloopsex9.
  _mlir_ciface_mlir_gccloopsex9(ret_ex9, &input_1, &input_2);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx9: MLIR GccLoopsEx9 Operation" << std::endl;
  std::cout << "[ ";
  std::cout << ret << " ";
  std::cout << "]" << std::endl;
}
