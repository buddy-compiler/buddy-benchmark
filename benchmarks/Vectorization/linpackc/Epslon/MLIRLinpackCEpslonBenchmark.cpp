//===- MLIRLinpackCEpslonBenchmark.cpp-------------------------------------===//
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
// This file implements the benchmark for epslon function.
//
//===----------------------------------------------------------------------===//

#include "Epslon.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>

// Declare the linpackcepslon C interface.
extern "C" {
float _mlir_ciface_mlir_linpackcepslonf32(float x);
double _mlir_ciface_mlir_linpackcepslonf64(double x);
}

// Define input and output sizes.
constexpr float x_input_f32 = 10.0;
constexpr double x_input_f64 = 10.0;

// Define the benchmark function.
static void MLIR_EpslonF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcepslonf32(x_input_f32);
    }
  }
}
static void MLIR_EpslonF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcepslonf64(x_input_f64);
    }
  }
}
// Define the benchmark function.
static void Epslon_float_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      epslon_float_gcc(x_input_f32);
    }
  }
}
static void Epslon_double_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      epslon_double_gcc(x_input_f64);
    }
  }
}
static void Epslon_float_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      epslon_float_clang(x_input_f32);
    }
  }
}
static void Epslon_double_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      epslon_double_clang(x_input_f64);
    }
  }
}
// Register benchmarking function.
BENCHMARK(MLIR_EpslonF32)->Arg(1);
BENCHMARK(MLIR_EpslonF64)->Arg(1);
BENCHMARK(Epslon_float_gcc)->Arg(1);
BENCHMARK(Epslon_double_gcc)->Arg(1);
BENCHMARK(Epslon_float_clang)->Arg(1);
BENCHMARK(Epslon_double_clang)->Arg(1);

// Generate result image.
void generateResultMLIRLinpackCEpslon() {
  // Define the MemRef descriptor for inputs and output.

  // Run the linpackcepslon.
  float x_f32 = _mlir_ciface_mlir_linpackcepslonf32(x_input_f32);
  double x_f64 = _mlir_ciface_mlir_linpackcepslonf64(x_input_f64);
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_LinpackC: MLIR epslon Operation for 'incx = 2'"
            << std::endl;
  std::cout << "f32: [ ";
  std::cout << x_f32;
  std::cout << "]" << std::endl;
  std::cout << "f64: [ ";
  std::cout << x_f64;
  std::cout << "]" << std::endl;
}
