//===- MLIRLinpackCMatgenBenchmark.cpp-------------------------------------===//
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
// This file implements the benchmark for matgen function.
//
//===----------------------------------------------------------------------===//

#include "Matgen.h"
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <cassert>
#include <iostream>
// Declare the linpackcdaxpy C interface.
extern "C" {
void _mlir_ciface_mlir_linpackcmatgenf32(MemRef<float, 1> *a, size_t lda,
                                         size_t n, MemRef<float, 1> *b,
                                         MemRef<float, 1> *norma);

void _mlir_ciface_mlir_linpackcmatgenf64(MemRef<double, 1> *a, size_t lda,
                                         size_t n, MemRef<double, 1> *b,
                                         MemRef<double, 1> *norma);
}

// Define input and output sizes.
intptr_t sizesArrayMLIRLinpackCMatgena[1] = {90000};
intptr_t sizesArrayMLIRLinpackCMatgenb[1] = {300};
intptr_t sizesArrayMLIRLinpackCMatgenNorma[1] = {1};
size_t lda = 3;
size_t n_matgen = 3;
// Define the MemRef container for inputs and output.
MemRef<float, 1> MLIRMatgen_af32(sizesArrayMLIRLinpackCMatgena, 0.0);
MemRef<float, 1> MLIRMatgen_bf32(sizesArrayMLIRLinpackCMatgenb, 0.0);
MemRef<float, 1> MLIRMatgen_normaf32(sizesArrayMLIRLinpackCMatgenNorma, 0.0);

MemRef<double, 1> MLIRMatgen_af64(sizesArrayMLIRLinpackCMatgena, 0.0);
MemRef<double, 1> MLIRMatgen_bf64(sizesArrayMLIRLinpackCMatgenb, 0.0);
MemRef<double, 1> MLIRMatgen_normaf64(sizesArrayMLIRLinpackCMatgenNorma, 0.0);

// Define the benchmark function.
static void MLIR_MatgenF32(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {

      _mlir_ciface_mlir_linpackcmatgenf32(&MLIRMatgen_af32, lda, n_matgen,
                                          &MLIRMatgen_bf32,
                                          &MLIRMatgen_normaf32);
    }
  }
}

static void MLIR_MatgenF64(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {

      _mlir_ciface_mlir_linpackcmatgenf64(&MLIRMatgen_af64, lda, n_matgen,
                                          &MLIRMatgen_bf64,
                                          &MLIRMatgen_normaf64);
    }
  }
}

static void Matgen_float_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      matgen_float_gcc(MLIRMatgen_af32.getData(), lda, n_matgen,
                       MLIRMatgen_bf32.getData(),
                       MLIRMatgen_normaf32.getData());
    }
  }
}

static void Matgen_double_gcc(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      matgen_double_gcc(MLIRMatgen_af64.getData(), lda, n_matgen,
                        MLIRMatgen_bf64.getData(),
                        MLIRMatgen_normaf64.getData());
    }
  }
}

static void Matgen_float_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      matgen_float_clang(MLIRMatgen_af32.getData(), lda, n_matgen,
                         MLIRMatgen_bf32.getData(),
                         MLIRMatgen_normaf32.getData());
    }
  }
}

static void Matgen_double_clang(benchmark::State &state) {
  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      matgen_double_clang(MLIRMatgen_af64.getData(), lda, n_matgen,
                          MLIRMatgen_bf64.getData(),
                          MLIRMatgen_normaf64.getData());
    }
  }
}
// Register benchmarking function.
BENCHMARK(MLIR_MatgenF32)->Arg(1);
BENCHMARK(MLIR_MatgenF64)->Arg(1);
BENCHMARK(Matgen_float_gcc)->Arg(1);
BENCHMARK(Matgen_double_gcc)->Arg(1);
BENCHMARK(Matgen_float_clang)->Arg(1);
BENCHMARK(Matgen_double_clang)->Arg(1);

// Generate result image.
void generateResultMLIRLinpackCMatgen() {

  // Define the MemRef container for inputs and output.
  MemRef<float, 1> MLIRMatgen_af32(sizesArrayMLIRLinpackCMatgena, 0.0);
  MemRef<float, 1> MLIRMatgen_bf32(sizesArrayMLIRLinpackCMatgenb, 0.0);
  MemRef<float, 1> MLIRMatgen_normaf32(sizesArrayMLIRLinpackCMatgenNorma, 0.0);

  MemRef<double, 1> MLIRMatgen_af64(sizesArrayMLIRLinpackCMatgena, 0.0);
  MemRef<double, 1> MLIRMatgen_bf64(sizesArrayMLIRLinpackCMatgenb, 0.0);
  MemRef<double, 1> MLIRMatgen_normaf64(sizesArrayMLIRLinpackCMatgenNorma, 0.0);
  // Run the linpackcdmatgen.
  _mlir_ciface_mlir_linpackcmatgenf32(&MLIRMatgen_af32, lda, n_matgen,
                                      &MLIRMatgen_bf32, &MLIRMatgen_normaf32);

  _mlir_ciface_mlir_linpackcmatgenf64(&MLIRMatgen_af64, lda, n_matgen,
                                      &MLIRMatgen_bf64, &MLIRMatgen_normaf64);

  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "MLIR_LinpackC: MLIR Matgen Operation" << std::endl;
  std::cout << "f32 a: [ ";
  for (size_t i = 0; i < MLIRMatgen_af32.getSize(); i++) {
    std::cout << MLIRMatgen_af32[i] << " ";
    assert(MLIRMatgen_af32.getData()[i] == a[i]);
  }
  std::cout << "]" << std::endl;

  std::cout << "f32 b: [ ";
  for (size_t i = 0; i < MLIRMatgen_bf32.getSize(); i++) {
    std::cout << MLIRMatgen_bf32.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << "f64 a: [ ";
  for (size_t i = 0; i < MLIRMatgen_af64.getSize(); i++) {
    std::cout << MLIRMatgen_af64.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << "f64 b: [ ";
  for (size_t i = 0; i < MLIRMatgen_bf64.getSize(); i++) {
    std::cout << MLIRMatgen_bf64.getData()[i] << " ";
  }
  std::cout << "]" << std::endl;
}
