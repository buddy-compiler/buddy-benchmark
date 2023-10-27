//===- LinpackCDdotBenchmark.cpp -----------------------------------------===//
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
#include "Ddot.h"
// Declare the linpackcddot C interface.
extern "C" {
float _mlir_ciface_mlir_linpackcddotrollf32(int n,
                                            MemRef<float, 1> *dx, int incx,
                                            MemRef<float, 1> *dy, int incy);
double _mlir_ciface_mlir_linpackcddotrollf64(int n,
                                            MemRef<double, 1> *dx, int incx,
                                            MemRef<double, 1> *dy, int incy); 
float _mlir_ciface_mlir_linpackcddotunrollf32(int n,
                                            MemRef<float, 1> *dx, int incx,
                                            MemRef<float, 1> *dy, int incy);
double _mlir_ciface_mlir_linpackcddotunrollf64(int n,
                                            MemRef<double, 1> *dx, int incx,
                                            MemRef<double, 1> *dy, int incy);                                                                                   
}

// Define input and output sizes.
constexpr int n = 10000;
constexpr int input_incx = -1;
constexpr int input_incy = 2;
constexpr int size_x = input_incx < 0 ? -input_incx : input_incx;
constexpr int size_y = input_incy < 0 ? -input_incy : input_incy;
constexpr int size = size_x < size_y ? size_y : size_x;

intptr_t sizesArrayMLIRLinpackCDdot[1] = {intptr_t(n * size)};
// Define the MemRef container for inputs and output.
MemRef<float, 1> inputMLIRDdot_dxf32(sizesArrayMLIRLinpackCDdot, 2.0);
MemRef<float, 1> inputMLIRDdot_dyf32(sizesArrayMLIRLinpackCDdot, 1.0);

MemRef<double, 1> inputMLIRDdot_dxf64(sizesArrayMLIRLinpackCDdot, 2.0);
MemRef<double, 1> inputMLIRDdot_dyf64(sizesArrayMLIRLinpackCDdot, 1.0);



// Define the benchmark function.
static void MLIR_DdotRollF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcddotrollf32(n, 
                                             &inputMLIRDdot_dxf32, input_incx,
                                             &inputMLIRDdot_dyf32, input_incy);
    }
  }
}
static void MLIR_DdotRollF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcddotrollf64(n, 
                                             &inputMLIRDdot_dxf64, input_incx,
                                             &inputMLIRDdot_dyf64, input_incy);
    }
  }
}
static void MLIR_DdotUnrollF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcddotunrollf32(n, 
                                             &inputMLIRDdot_dxf32, input_incx,
                                             &inputMLIRDdot_dyf32, input_incy);
    }
  }
}
static void MLIR_DdotUnrollF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      _mlir_ciface_mlir_linpackcddotunrollf64(n, 
                                             &inputMLIRDdot_dxf64, input_incx,
                                             &inputMLIRDdot_dyf64, input_incy);
    }
  }
}

static void Ddot_ROLL_float_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      ddot_ROLL_float_gcc(n,inputMLIRDdot_dxf32.getData(),input_incx,inputMLIRDdot_dyf32.getData(),input_incy);
    }
  }
}

static void Ddot_ROLL_double_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      ddot_ROLL_double_gcc(n,inputMLIRDdot_dxf64.getData(),input_incx,inputMLIRDdot_dyf64.getData(),input_incy);
    }
  }
}

static void Ddot_UNROLL_float_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      ddot_UNROLL_float_gcc(n,inputMLIRDdot_dxf32.getData(),input_incx,inputMLIRDdot_dyf32.getData(),input_incy);
    }
  }
}

static void Ddot_UNROLL_double_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      ddot_UNROLL_double_gcc(n,inputMLIRDdot_dxf64.getData(),input_incx,inputMLIRDdot_dyf64.getData(),input_incy);
    }
  }
}

//clang
static void Ddot_ROLL_float_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      ddot_ROLL_float_clang(n,inputMLIRDdot_dxf32.getData(),input_incx,inputMLIRDdot_dyf32.getData(),input_incy);
    }
  }
}

static void Ddot_ROLL_double_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      ddot_ROLL_double_clang(n,inputMLIRDdot_dxf64.getData(),input_incx,inputMLIRDdot_dyf64.getData(),input_incy);
    }
  }
}

static void Ddot_UNROLL_float_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      ddot_UNROLL_float_clang(n,inputMLIRDdot_dxf32.getData(),input_incx,inputMLIRDdot_dyf32.getData(),input_incy);
    }
  }
}

static void Ddot_UNROLL_double_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      ddot_UNROLL_double_clang(n,inputMLIRDdot_dxf64.getData(),input_incx,inputMLIRDdot_dyf64.getData(),input_incy);
    }
  }
}
// Register benchmarking function.
BENCHMARK(MLIR_DdotRollF32)->Arg(1);
BENCHMARK(MLIR_DdotRollF64)->Arg(1);
BENCHMARK(MLIR_DdotUnrollF32)->Arg(1);
BENCHMARK(MLIR_DdotUnrollF64)->Arg(1);
BENCHMARK(Ddot_ROLL_float_gcc)->Arg(1);
BENCHMARK(Ddot_ROLL_double_gcc)->Arg(1);
BENCHMARK(Ddot_UNROLL_float_gcc)->Arg(1);
BENCHMARK(Ddot_UNROLL_double_gcc)->Arg(1);
BENCHMARK(Ddot_ROLL_float_clang)->Arg(1);
BENCHMARK(Ddot_ROLL_double_clang)->Arg(1);
BENCHMARK(Ddot_UNROLL_float_clang)->Arg(1);
BENCHMARK(Ddot_UNROLL_double_clang)->Arg(1);
// Generate result image.
void generateResultMLIRLinpackCDdot() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<float, 1> inputMLIRDdot_dxf32(sizesArrayMLIRLinpackCDdot, 2.0);
  MemRef<float, 1> inputMLIRDdot_dyf32_roll(sizesArrayMLIRLinpackCDdot, 1.0);
  MemRef<float, 1> inputMLIRDdot_dyf32_unroll(sizesArrayMLIRLinpackCDdot, 1.0);
  MemRef<double, 1> inputMLIRDdot_dxf64(sizesArrayMLIRLinpackCDdot, 2.0);
  MemRef<double, 1> inputMLIRDdot_dyf64_roll(sizesArrayMLIRLinpackCDdot, 1.0);
  MemRef<double, 1> inputMLIRDdot_dyf64_unroll(sizesArrayMLIRLinpackCDdot,1.0);
  // Run the linpackcddot.
  float ddot_res1;
  ddot_res1 = _mlir_ciface_mlir_linpackcddotrollf32(n, 
                                             &inputMLIRDdot_dxf32, input_incx,
                                             &inputMLIRDdot_dyf32_roll, input_incy);
  double ddot_res2;
  ddot_res2 = _mlir_ciface_mlir_linpackcddotrollf64(n, 
                                             &inputMLIRDdot_dxf64, input_incx,
                                             &inputMLIRDdot_dyf64_roll, input_incy);                                           
  float ddot_res3;
  ddot_res3 = _mlir_ciface_mlir_linpackcddotunrollf32(n, 
                                             &inputMLIRDdot_dxf32, input_incx,
                                             &inputMLIRDdot_dyf32_unroll, input_incy);
  double ddot_res4;
  ddot_res4 = _mlir_ciface_mlir_linpackcddotunrollf64(n, 
                                             &inputMLIRDdot_dxf64, input_incx,
                                             &inputMLIRDdot_dyf64_unroll, input_incy);  
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_LinpackC: MLIR Ddot Operation for 'incx = -1, incy = 2'"
            << std::endl;
  std::cout << "f32roll: [ ";
  for (size_t i = 0; i < sizesArrayMLIRLinpackCDdot[0]; i++) {
    std::cout << inputMLIRDdot_dxf32.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "ddot_res: "<< ddot_res1 << std::endl;
  std::cout << "f64roll: [ ";
  for (size_t i = 0; i < sizesArrayMLIRLinpackCDdot[0]; i++) {
    std::cout << inputMLIRDdot_dxf64.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
  std::cout << "ddot_res: "<< ddot_res2 << std::endl;
  std::cout << "ddot_res: "<< ddot_res3 << std::endl;
  std::cout << "ddot_res: "<< ddot_res4 << std::endl;
}
