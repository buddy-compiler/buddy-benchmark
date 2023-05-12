//===- LinpackCDaxpyBenchmark.cpp -----------------------------------------===//
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

// Declare the linpackcdaxpy C interface.
extern "C" {
void _mlir_ciface_mlir_linpackcdaxpyrollf32(size_t n, float da,
                                            MemRef<float, 1> *dx, size_t incx,
                                            MemRef<float, 1> *dy, size_t incy);
void _mlir_ciface_mlir_linpackcdaxpyrollf64(size_t n, double da,
                                            MemRef<double, 1> *dx, size_t incx,
                                            MemRef<double, 1> *dy, size_t incy);
void _mlir_ciface_mlir_linpackcdaxpyunrollf32(size_t n, float da,
                                              MemRef<float, 1> *dx, size_t incx,
                                              MemRef<float, 1> *dy,
                                              size_t incy);
void _mlir_ciface_mlir_linpackcdaxpyunrollf64(size_t n, double da,
                                              MemRef<double, 1> *dx,
                                              size_t incx,
                                              MemRef<double, 1> *dy,
                                              size_t incy);
}

// Define input and output sizes.
intptr_t sizesArrayMLIRLinpackCDaxpy[1] = {10};
size_t n = 10;
size_t input_incx = 1;
size_t input_incy = 1;
// Define the MemRef container for inputs and output.
float input_dx_f32[10] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1, 10.2};
float input_da_f32 = 10.3;
MemRef<float, 1> inputMLIRDaxpy_dxf32(input_dx_f32,
                                      sizesArrayMLIRLinpackCDaxpy);
MemRef<float, 1> outputMLIRDaxpy_f32(sizesArrayMLIRLinpackCDaxpy, 0.0);

double input_dx_f64[10] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1, 10.2};
double input_da_f64 = 10.3;
MemRef<double, 1> inputMLIRDaxpy_dxf64(input_dx_f64,
                                       sizesArrayMLIRLinpackCDaxpy);
MemRef<double, 1> outputMLIRDaxpy_f64(sizesArrayMLIRLinpackCDaxpy, 0.0);

// Define the benchmark function.
static void MLIR_DaxpyRollF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdaxpyrollf32(n, input_da_f32,
                                             &inputMLIRDaxpy_dxf32, input_incx,
                                             &outputMLIRDaxpy_f32, input_incy);
    }
  }
}

static void MLIR_DaxpyRollF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdaxpyrollf64(n, input_da_f64,
                                             &inputMLIRDaxpy_dxf64, input_incx,
                                             &outputMLIRDaxpy_f64, input_incy);
    }
  }
}

static void MLIR_DaxpyUnrollF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdaxpyunrollf32(
          n, input_da_f32, &inputMLIRDaxpy_dxf32, input_incx,
          &outputMLIRDaxpy_f32, input_incy);
    }
  }
}

static void MLIR_DaxpyUnrollF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcdaxpyunrollf64(
          n, input_da_f64, &inputMLIRDaxpy_dxf64, input_incx,
          &outputMLIRDaxpy_f64, input_incy);
    }
  }
}

// Register benchmarking function.
BENCHMARK(MLIR_DaxpyRollF32)->Arg(1);
BENCHMARK(MLIR_DaxpyRollF64)->Arg(1);
BENCHMARK(MLIR_DaxpyUnrollF32)->Arg(1);
BENCHMARK(MLIR_DaxpyUnrollF64)->Arg(1);

// Generate result image.
void generateResultMLIRLinpackCDaxpy() {
  // Define the MemRef descriptor for inputs and output.
  float input_dx_f32[10] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1, 10.2};
  float input_da_f32 = 10.3;
  MemRef<float, 1> inputMLIRDaxpy_dxf32(input_dx_f32,
                                        sizesArrayMLIRLinpackCDaxpy);
  MemRef<float, 1> outputMLIRDaxpy_f32_roll(sizesArrayMLIRLinpackCDaxpy, 0.0);
  MemRef<float, 1> outputMLIRDaxpy_f32_unroll(sizesArrayMLIRLinpackCDaxpy, 0.0);

  double input_dx_f64[10] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1, 10.2};
  double input_da_f64 = 10.3;
  MemRef<double, 1> inputMLIRDaxpy_dxf64(input_dx_f64,
                                         sizesArrayMLIRLinpackCDaxpy);
  MemRef<double, 1> outputMLIRDaxpy_f64_roll(sizesArrayMLIRLinpackCDaxpy, 0.0);
  MemRef<double, 1> outputMLIRDaxpy_f64_unroll(sizesArrayMLIRLinpackCDaxpy,
                                               0.0);
  // Run the linpackcdaxpy.
  _mlir_ciface_mlir_linpackcdaxpyrollf32(n, input_da_f32, &inputMLIRDaxpy_dxf32,
                                         input_incx, &outputMLIRDaxpy_f32_roll,
                                         input_incy);
  _mlir_ciface_mlir_linpackcdaxpyunrollf32(
      n, input_da_f32, &inputMLIRDaxpy_dxf32, input_incx,
      &outputMLIRDaxpy_f32_unroll, input_incy);
  _mlir_ciface_mlir_linpackcdaxpyrollf64(n, input_da_f64, &inputMLIRDaxpy_dxf64,
                                         input_incx, &outputMLIRDaxpy_f64_roll,
                                         input_incy);
  _mlir_ciface_mlir_linpackcdaxpyunrollf64(
      n, input_da_f64, &inputMLIRDaxpy_dxf64, input_incx,
      &outputMLIRDaxpy_f64_unroll, input_incy);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_LinpackC: MLIR Daxpy Operation" << std::endl;
  std::cout << "f32roll: [ ";
  for (size_t i = 0; i < outputMLIRDaxpy_f32_roll.getSize(); i++) {
    std::cout << outputMLIRDaxpy_f32_roll.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << "f32unroll: [ ";
  for (size_t i = 0; i < outputMLIRDaxpy_f32_unroll.getSize(); i++) {
    std::cout << outputMLIRDaxpy_f32_unroll.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << "f64roll: [ ";
  for (size_t i = 0; i < outputMLIRDaxpy_f64_roll.getSize(); i++) {
    std::cout << outputMLIRDaxpy_f64_roll.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << "f64unroll: [ ";
  for (size_t i = 0; i < outputMLIRDaxpy_f64_unroll.getSize(); i++) {
    std::cout << outputMLIRDaxpy_f64_unroll.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
