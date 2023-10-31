//===- MLIRLinpackCDscalBenchmark.cpp ----------------------------------- -===//
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
// This file implements the benchmark for dscal function.
//
//===----------------------------------------------------------------------===//

#include "Dscal.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
// Declare the linpackcdscal C interface.
extern "C" {
void _mlir_ciface_mlir_linpackcdscalrollf32(int n, float da,
                                            MemRef<float, 1> *dx, int incx);
void _mlir_ciface_mlir_linpackcdscalunrollf32(int n, float da,
                                              MemRef<float, 1> *dx, int incx);
void _mlir_ciface_mlir_linpackcdscalrollf64(int n, double da,
                                            MemRef<double, 1> *dx, int incx);
void _mlir_ciface_mlir_linpackcdscalunrollf64(int n, double da,
                                              MemRef<double, 1> *dx, int incx);
}

// Define input and output sizes.
constexpr int n = 10000;
constexpr int input_incx = 2;

intptr_t sizesArrayMLIRLinpackCDscal[1] = {intptr_t(n * input_incx)};
// Define the MemRef container for inputs and output.
float input_dscal_da_f32 = 10.3;
MemRef<float, 1> inputMLIRDscal_dxf32(sizesArrayMLIRLinpackCDscal, 2.3);

double input_dscal_da_f64 = 10.3;
MemRef<double, 1> inputMLIRDscal_dxf64(sizesArrayMLIRLinpackCDscal, 2.3);

// Define the benchmark function.
static void MLIR_DscalRollF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdscalrollf32(n, input_dscal_da_f32,
                                             &inputMLIRDscal_dxf32, input_incx);
    }
  }
}

static void MLIR_DscalUnrollF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdscalunrollf32(
          n, input_dscal_da_f32, &inputMLIRDscal_dxf32, input_incx);
    }
  }
}
static void MLIR_DscalRollF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdscalrollf64(n, input_dscal_da_f64,
                                             &inputMLIRDscal_dxf64, input_incx);
    }
  }
}
static void MLIR_DscalUnrollF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdscalrollf64(n, input_dscal_da_f64,
                                             &inputMLIRDscal_dxf64, input_incx);
    }
  }
}

static void Dscal_Roll_float_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dscal_ROLL_float_gcc(n, input_dscal_da_f32,
                           inputMLIRDscal_dxf32.getData(), input_incx);
    }
  }
}

static void Dscal_Unroll_float_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dscal_UNROLL_float_gcc(n, input_dscal_da_f32,
                             inputMLIRDscal_dxf32.getData(), input_incx);
    }
  }
}
static void Dscal_Roll_double_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dscal_ROLL_double_gcc(n, input_dscal_da_f64,
                            inputMLIRDscal_dxf64.getData(), input_incx);
    }
  }
}
static void Dscal_Unroll_double_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dscal_UNROLL_double_gcc(n, input_dscal_da_f64,
                              inputMLIRDscal_dxf64.getData(), input_incx);
    }
  }
}
// clang
static void Dscal_Roll_float_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dscal_ROLL_float_clang(n, input_dscal_da_f32,
                             inputMLIRDscal_dxf32.getData(), input_incx);
    }
  }
}

static void Dscal_Unroll_float_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dscal_UNROLL_float_clang(n, input_dscal_da_f32,
                               inputMLIRDscal_dxf32.getData(), input_incx);
    }
  }
}
static void Dscal_Roll_double_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dscal_ROLL_double_clang(n, input_dscal_da_f64,
                              inputMLIRDscal_dxf64.getData(), input_incx);
    }
  }
}
static void Dscal_Unroll_double_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      dscal_UNROLL_double_clang(n, input_dscal_da_f64,
                                inputMLIRDscal_dxf64.getData(), input_incx);
    }
  }
}
// Register benchmarking function.
BENCHMARK(MLIR_DscalRollF32)->Arg(1);
BENCHMARK(MLIR_DscalUnrollF32)->Arg(1);
BENCHMARK(MLIR_DscalRollF64)->Arg(1);
BENCHMARK(MLIR_DscalUnrollF64)->Arg(1);

BENCHMARK(Dscal_Roll_float_gcc)->Arg(1);
BENCHMARK(Dscal_Roll_double_gcc)->Arg(1);
BENCHMARK(Dscal_Unroll_float_gcc)->Arg(1);
BENCHMARK(Dscal_Unroll_double_gcc)->Arg(1);

BENCHMARK(Dscal_Roll_float_clang)->Arg(1);
BENCHMARK(Dscal_Roll_double_clang)->Arg(1);
BENCHMARK(Dscal_Unroll_float_clang)->Arg(1);
BENCHMARK(Dscal_Unroll_double_clang)->Arg(1);
// Generate result image.
void generateResultMLIRLinpackCDscal() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<float, 1> inputMLIRDscal_dxf32(sizesArrayMLIRLinpackCDscal, 2.3);
  MemRef<float, 1> inputMLIRDscal_f32_roll(sizesArrayMLIRLinpackCDscal, 2.3);
  MemRef<float, 1> inputMLIRDscal_f32_unroll(sizesArrayMLIRLinpackCDscal, 2.3);
  MemRef<double, 1> inputMLIRDscal_dxf64(sizesArrayMLIRLinpackCDscal, 2.3);
  MemRef<double, 1> inputMLIRDscal_f64_roll(sizesArrayMLIRLinpackCDscal, 2.3);
  MemRef<double, 1> inputMLIRDscal_f64_unroll(sizesArrayMLIRLinpackCDscal, 2.3);
  // Run the linpackcdscal.
  _mlir_ciface_mlir_linpackcdscalrollf32(n, input_dscal_da_f32,
                                         &inputMLIRDscal_dxf32, input_incx);

  _mlir_ciface_mlir_linpackcdscalunrollf32(
      n, input_dscal_da_f32, &inputMLIRDscal_f32_unroll, input_incx);

  _mlir_ciface_mlir_linpackcdscalrollf64(n, input_dscal_da_f64,
                                         &inputMLIRDscal_dxf64, input_incx);

  _mlir_ciface_mlir_linpackcdscalunrollf64(
      n, input_dscal_da_f64, &inputMLIRDscal_f64_unroll, input_incx);
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_LinpackC: MLIR Dscal Operation for 'incx = 2'"
            << std::endl;
  std::cout << "f32roll: [ ";
  for (size_t i = 0; i < n * input_incx; i++) {
    std::cout << inputMLIRDscal_dxf32.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "f32unroll: [ ";
  for (size_t i = 0; i < n * input_incx; i++) {
    std::cout << inputMLIRDscal_f32_unroll.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "f64roll: [ ";
  for (size_t i = 0; i < n * input_incx; i++) {
    std::cout << inputMLIRDscal_dxf64.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "f64unroll: [ ";
  for (size_t i = 0; i < n * input_incx; i++) {
    std::cout << inputMLIRDscal_f64_unroll.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
