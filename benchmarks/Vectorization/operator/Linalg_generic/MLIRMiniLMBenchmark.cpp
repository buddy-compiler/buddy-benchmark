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
void _mlir_ciface_mlir_minilm_1();
void _mlir_ciface_mlir_minilm_2();
void _mlir_ciface_mlir_minilm_3();
void _mlir_ciface_mlir_minilm_4();
void _mlir_ciface_mlir_minilm_5();
void _mlir_ciface_mlir_minilm_6();
void _mlir_ciface_mlir_minilm_7();
}

static void MLIR_MiniLM_1(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_minilm_1();
    }
  }
}

static void MLIR_MiniLM_2(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_minilm_2();
    }
  }
}

static void MLIR_MiniLM_3(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_minilm_3();
    }
  }
}
static void MLIR_MiniLM_4(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_minilm_4();
    }
  }
}
static void MLIR_MiniLM_5(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_minilm_5();
    }
  }
}

static void MLIR_MiniLM_6(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_minilm_6();
    }
  }
}
static void MLIR_MiniLM_7(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_minilm_7();
    }
  }
}
// Register benchmarking function.
BENCHMARK(MLIR_MiniLM_1)->Arg(1);
BENCHMARK(MLIR_MiniLM_2)->Arg(1);
BENCHMARK(MLIR_MiniLM_3)->Arg(1);
BENCHMARK(MLIR_MiniLM_4)->Arg(1);
BENCHMARK(MLIR_MiniLM_5)->Arg(1);
BENCHMARK(MLIR_MiniLM_6)->Arg(1);
BENCHMARK(MLIR_MiniLM_7)->Arg(1);
// Generate result image.
//void generateResultMLIRMatVec() {
  // // Define the MemRef descriptor for input1, intput2, and output.
  // // Run the 2D matvec.
  // _mlir_ciface_mlir_matvec(&input1, &input2, &output);
  // // Print the output.
  // std::cout << "--------------------------------------------------------"
  //           << std::endl;
  // std::cout << "MLIR_MatVec: MLIR MatVec Operation" << std::endl;
  // std::cout << "[ ";
  // for (size_t i = 0; i < output.getSize(); i++) {
  //   std::cout << output.getData()[i] << " ";
  // }
  // std::cout << "]" << std::endl;
//}
