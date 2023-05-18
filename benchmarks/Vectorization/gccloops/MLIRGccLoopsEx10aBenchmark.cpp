//===- MLIRGccLoopsEx10aBenchmark.cpp --------------------------------------------===//
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

// Declare the gccloopsex10a C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex10a(MemRef<short, 1> *sa, MemRef<short, 1> *sb, MemRef<short, 1> *sc,
                                     MemRef<int, 1> *ia, MemRef<int, 1> *ib, MemRef<int, 1> *ic);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx10a[1] = {10};
intptr_t sizesOutputArrayMLIRGccLoopsEx10a[1] = {10};
// Define the MemRef container for inputs and outputs.
int input1i_data_ex10a[10] = {1,2,3,4,5,6,7,8,9,10};
int input2i_data_ex10a[10] = {1,1,1,1,1,1,1,1,1,1};
short input1s_data_ex10a[10] = {1,2,3,4,5,6,7,8,9,10};
short input2s_data_ex10a[10] = {1,1,1,1,1,1,1,1,1,1};
MemRef<int, 1> input1iMLIRGccLoopsEx10a(input1i_data_ex10a, sizesInputArrayMLIRGccLoopsEx10a);
MemRef<int, 1> input2iMLIRGccLoopsEx10a(input2i_data_ex10a, sizesInputArrayMLIRGccLoopsEx10a);
MemRef<int, 1> outputiMLIRGccLoopsEx10a(sizesOutputArrayMLIRGccLoopsEx10a, 0);
MemRef<short, 1> input1sMLIRGccLoopsEx10a(input1s_data_ex10a, sizesInputArrayMLIRGccLoopsEx10a);
MemRef<short, 1> input2sMLIRGccLoopsEx10a(input2s_data_ex10a, sizesInputArrayMLIRGccLoopsEx10a);
MemRef<short, 1> outputsMLIRGccLoopsEx10a(sizesOutputArrayMLIRGccLoopsEx10a, 0);

static void MLIR_GccLoopsEx10a(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_gccloopsex10a(&outputsMLIRGccLoopsEx10a, &input1sMLIRGccLoopsEx10a, &input2sMLIRGccLoopsEx10a, 
                                      &outputiMLIRGccLoopsEx10a, &input1iMLIRGccLoopsEx10a, &input2iMLIRGccLoopsEx10a);
    }
  }
} 

// Register benchmarking function.
BENCHMARK(MLIR_GccLoopsEx10a)->Arg(1);

// Generate result image.
void generateResultMLIRGccLoopsEx10a() {
  // Define the MemRef descriptor for inputs and outputs.
  int input1i_data_ex10a[10] = {1,2,3,4,5,6,7,8,9,10};
  int input2i_data_ex10a[10] = {1,1,1,1,1,1,1,1,1,1};
  short input1s_data_ex10a[10] = {2,3,4,5,6,7,8,9,10,11};
  short input2s_data_ex10a[10] = {1,1,1,1,1,1,1,1,1,1};
  MemRef<int, 1> input1i(input1i_data_ex10a, sizesInputArrayMLIRGccLoopsEx10a);
  MemRef<int, 1> input2i(input2i_data_ex10a, sizesInputArrayMLIRGccLoopsEx10a);
  MemRef<int, 1> outputi(sizesOutputArrayMLIRGccLoopsEx10a, 0);
  MemRef<short, 1> input1s(input1s_data_ex10a, sizesInputArrayMLIRGccLoopsEx10a);
  MemRef<short, 1> input2s(input2s_data_ex10a, sizesInputArrayMLIRGccLoopsEx10a);
  MemRef<short, 1> outputs(sizesOutputArrayMLIRGccLoopsEx10a, 0);
  // Run the gccloopsex10a.
  _mlir_ciface_mlir_gccloopsex10a(&outputs, &input1s, &input2s, 
                                  &outputi, &input1i, &input2i);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_GccLoopsEx10a: MLIR GccLoopsEx10a Operation" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < outputi.getSize(); i++) {
    std::cout << outputi.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < outputs.getSize(); i++) {
    std::cout << outputs.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
