//===- MLIRMatMulBenchmark.cpp --------------------------------------------===//
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

// Declare the matmul C interface.
extern "C" {
void _mlir_ciface_mlir_matmul(MemRef<float, 2> *input1,
                              MemRef<float, 2> *input2,
                              MemRef<float, 2> *output);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRMatMul1[2] = {5, 3};
intptr_t sizesInputArrayMLIRMatMul2[2] = {3, 2};
intptr_t sizesOutputArrayMLIRMatMul[2] = {5, 2};
// Define the MemRef container for input1, input2, and output.
MemRef<float, 2> inputMLIRMatMul1(sizesInputArrayMLIRMatMul1, 2);
MemRef<float, 2> inputMLIRMatMul2(sizesInputArrayMLIRMatMul2, 3);
MemRef<float, 2> outputMLIRMatMul(sizesOutputArrayMLIRMatMul, 0);

static void MLIR_MatMul(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_matmul(&inputMLIRMatMul1, &inputMLIRMatMul2,
                               &outputMLIRMatMul);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_MatMul)->Arg(1);

// Generate result image.
void generateResultMLIRMatMul() {
  // Define the MemRef descriptor for input1, intput2, and output.
  MemRef<float, 2> input1(sizesInputArrayMLIRMatMul1, 2);
  MemRef<float, 2> input2(sizesInputArrayMLIRMatMul2, 3);
  MemRef<float, 2> output(sizesOutputArrayMLIRMatMul, 0);
  // Run the 2D matmul.
  _mlir_ciface_mlir_matmul(&input1, &input2, &output);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_MatMul: MLIR MatMul Operation + Nested Loop" << std::endl;
  std::cout << "[ ";
  for (size_t i = 0; i < output.getSize(); i++) {
    std::cout << output.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
