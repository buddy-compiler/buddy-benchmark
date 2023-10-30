//===- MLIRLinpackCDmxpyBenchmark.cpp -------------------------------------===//
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
// This file implements the benchmark for dmxpy function.
//
//===----------------------------------------------------------------------===//

#include "Dmxpy.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
// Declare the linpackcdmxpy C interface.
extern "C" {
void _mlir_ciface_mlir_linpackcdmxpyf32(int n1, MemRef<float, 1> *y, int n2,
                                        int ldm, MemRef<float, 1> *x,
                                        MemRef<float, 1> *m);
void _mlir_ciface_mlir_linpackcdmxpyf64(int n1, MemRef<double, 1> *y, int n2,
                                        int ldm, MemRef<double, 1> *x,
                                        MemRef<double, 1> *m);
}

// Define input and output sizes.
constexpr int n1 = 90;
constexpr int n2 = 90;
constexpr int ldm = 90;

intptr_t sizesArrayMLIRLinpackCDmxpy[1] = {intptr_t(n1)};
// Define the MemRef container for inputs and output.
MemRef<float, 1> inputMLIRDmxpy_xf32(sizesArrayMLIRLinpackCDmxpy, 1.0);
MemRef<float, 1> inputMLIRDmxpy_yf32(sizesArrayMLIRLinpackCDmxpy, 2.0);
MemRef<float, 1> inputMLIRDmxpy_mf32(sizesArrayMLIRLinpackCDmxpy, 3.0);

MemRef<double, 1> inputMLIRDmxpy_xf64(sizesArrayMLIRLinpackCDmxpy, 1.0);
MemRef<double, 1> inputMLIRDmxpy_yf64(sizesArrayMLIRLinpackCDmxpy, 2.0);
MemRef<double, 1> inputMLIRDmxpy_mf64(sizesArrayMLIRLinpackCDmxpy, 3.0);

// Define the benchmark function.
static void MLIR_DmxpyF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdmxpyf32(n1, &inputMLIRDmxpy_yf32, n2, ldm,
                                         &inputMLIRDmxpy_xf32,
                                         &inputMLIRDmxpy_mf32);
    }
  }
}

static void MLIR_DmxpyF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdmxpyf64(n1, &inputMLIRDmxpy_yf64, n2, ldm,
                                         &inputMLIRDmxpy_xf64,
                                         &inputMLIRDmxpy_mf64);
    }
  }
}
static void Dmxpy_float_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dmxpy_float_gcc(n1, inputMLIRDmxpy_yf32.getData(), n2, ldm,
                      inputMLIRDmxpy_xf32.getData(),
                      inputMLIRDmxpy_mf32.getData());
    }
  }
}
static void Dmxpy_double_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dmxpy_double_gcc(n1, inputMLIRDmxpy_yf64.getData(), n2, ldm,
                       inputMLIRDmxpy_xf64.getData(),
                       inputMLIRDmxpy_mf64.getData());
    }
  }
}

static void Dmxpy_float_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dmxpy_float_clang(n1, inputMLIRDmxpy_yf32.getData(), n2, ldm,
                        inputMLIRDmxpy_xf32.getData(),
                        inputMLIRDmxpy_mf32.getData());
    }
  }
}
static void Dmxpy_double_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dmxpy_double_clang(n1, inputMLIRDmxpy_yf64.getData(), n2, ldm,
                         inputMLIRDmxpy_xf64.getData(),
                         inputMLIRDmxpy_mf64.getData());
    }
  }
}
// Register benchmarking function.
BENCHMARK(MLIR_DmxpyF32)->Arg(1);
BENCHMARK(MLIR_DmxpyF64)->Arg(1);

BENCHMARK(Dmxpy_float_gcc)->Arg(1);
BENCHMARK(Dmxpy_double_gcc)->Arg(1);

BENCHMARK(Dmxpy_float_clang)->Arg(1);
BENCHMARK(Dmxpy_double_clang)->Arg(1);
// Generate result image.
void generateResultMLIRLinpackCDmxpy() {
  // Define the MemRef descriptor for inputs and output.

  // Run the linpackcdmxpy.
  _mlir_ciface_mlir_linpackcdmxpyf32(n1, &inputMLIRDmxpy_yf32, n2, ldm,
                                     &inputMLIRDmxpy_xf32,
                                     &inputMLIRDmxpy_mf32);

  _mlir_ciface_mlir_linpackcdmxpyf64(n1, &inputMLIRDmxpy_yf64, n2, ldm,
                                     &inputMLIRDmxpy_xf64,
                                     &inputMLIRDmxpy_mf64);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_LinpackC: MLIR Dmxpy Operation " << std::endl;
  std::cout << "yf32: [ ";
  for (size_t i = 0; i < n1; i++) {
    std::cout << inputMLIRDmxpy_yf32.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "xf32: [ ";
  for (size_t i = 0; i < n1; i++) {
    std::cout << inputMLIRDmxpy_xf32.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "mf32: [ ";
  for (size_t i = 0; i < n1; i++) {
    std::cout << inputMLIRDmxpy_mf32.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "f64: [ ";
  for (size_t i = 0; i < n1; i++) {
    std::cout << inputMLIRDmxpy_mf64.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
