//===- MLIRLinpackCIdamaxBenchmark.cpp-------------------------------------===//
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
// This file implements the benchmark for idamax function.
//
//===----------------------------------------------------------------------===//

#include "Idamax.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>

// Declare the linpackcidamax C interface.
extern "C" {
int _mlir_ciface_mlir_linpackcidamaxf32(int n, MemRef<float, 1> *dx, int incx);
int _mlir_ciface_mlir_linpackcidamaxf64(int n, MemRef<double, 1> *dx, int incx);
}

// Define input and output sizes.
constexpr int n = 10000;
constexpr int input_incx = 2;
float data[n * input_incx] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
double data2[n * input_incx] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
intptr_t sizesArrayMLIRLinpackCidamax[1] = {intptr_t(n * input_incx)};
// Define the MemRef container for inputs and output.
MemRef<float, 1> inputMLIRidamax_dxf32(data, sizesArrayMLIRLinpackCidamax, 0);

MemRef<double, 1> inputMLIRidamax_dxf64(data2, sizesArrayMLIRLinpackCidamax, 0);

// Define the benchmark function.
static void MLIR_idamaxF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcidamaxf32(n, &inputMLIRidamax_dxf32,
                                          input_incx);
    }
  }
}
static void MLIR_idamaxF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcidamaxf64(n, &inputMLIRidamax_dxf64,
                                          input_incx);
    }
  }
}

static void idamax_float_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      idamax_float_gcc(n, inputMLIRidamax_dxf32.getData(), input_incx);
    }
  }
}
static void idamax_double_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      idamax_double_gcc(n, inputMLIRidamax_dxf64.getData(), input_incx);
    }
  }
}

static void idamax_float_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      idamax_float_clang(n, inputMLIRidamax_dxf32.getData(), input_incx);
    }
  }
}
static void idamax_double_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      idamax_double_clang(n, inputMLIRidamax_dxf64.getData(), input_incx);
    }
  }
}
// Register benchmarking function.
BENCHMARK(MLIR_idamaxF32)->Arg(1);
BENCHMARK(MLIR_idamaxF64)->Arg(1);
BENCHMARK(idamax_float_gcc)->Arg(1);
BENCHMARK(idamax_double_gcc)->Arg(1);
BENCHMARK(idamax_float_clang)->Arg(1);
BENCHMARK(idamax_double_clang)->Arg(1);

// Generate result image.
void generateResultMLIRLinpackCIdamax() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<float, 1> inputMLIRidamax_dxf32(data, sizesArrayMLIRLinpackCidamax, 0);
  MemRef<double, 1> inputMLIRidamax_dxf64(data2, sizesArrayMLIRLinpackCidamax,
                                          0);
  // Run the linpackcidamax.
  int itemp = _mlir_ciface_mlir_linpackcidamaxf32(n, &inputMLIRidamax_dxf32,
                                                  input_incx);
  int itemp2 = _mlir_ciface_mlir_linpackcidamaxf64(n, &inputMLIRidamax_dxf64,
                                                   input_incx);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_LinpackC: MLIR idamax Operation for 'incx = 2'"
            << std::endl;
  std::cout << "f32: [ ";
  for (size_t i = 0; i < n * input_incx; i++) {
    std::cout << inputMLIRidamax_dxf32.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "itemp " << itemp << " " << std::endl;
  std::cout << "f64: [ ";
  for (size_t i = 0; i < n * input_incx; i++) {
    std::cout << inputMLIRidamax_dxf64.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "itemp " << itemp2 << " " << std::endl;
}
