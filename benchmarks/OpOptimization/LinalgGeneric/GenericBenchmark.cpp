//===- GenericBenchmark.cpp ----------------------------------------===//
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
// This file implements the benchmark for Linalg Generic operation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>
#include <random>

// Define target layout.
#define INPUT_H 4096
#define INPUT_W 4096
#define OUTPUT_W 4096

namespace {

// Helper functions and variables.
const std::string PASS = "\033[32mPASS\033[0m";
const std::string FAIL = "\033[32mFAIL\033[0m";
bool areArraysEqual(float array1[], float array2[], int size) {
  for (int i = 0; i < size; ++i)
  {
    if (array1[i] != array2[i]) {
      return false;
    }
  }
  return true;
}

// Declare the C interface.
extern "C" {
void _mlir_ciface_origin(MemRef<float, 2> *inpit, MemRef<float, 1> *outpit);
void _mlir_ciface_transform_tiling(MemRef<float, 2> *inpit, MemRef<float, 1> *outpit);
void _mlir_ciface_manul(MemRef<float, 2> *inpit, MemRef<float, 1> *outpit);
void _mlir_ciface_manul2(MemRef<float, 2> *inpit, MemRef<float, 1> *outpit);
void _mlir_ciface_manul3(MemRef<float, 2> *inpit, MemRef<float, 1> *outpit);
}

#define DEFINE_BENCHMARK(name, func)                  \
  void BM_GENERIC_##name(benchmark::State &state) {   \
    intptr_t sizesInput[2] = {INPUT_H, INPUT_W};      \
    intptr_t sizesOutput[1] = { OUTPUT_W};   \
    MemRef<float, 2> input(sizesInput, 1.0);               \
    MemRef<float, 1> output(sizesOutput, 1.0);             \
    for (auto _ : state) {                            \
      func(&input, &output);                          \
    }                                                 \
  }


DEFINE_BENCHMARK(ORIGIN, _mlir_ciface_origin)
DEFINE_BENCHMARK(TRANSFORM_TILING, _mlir_ciface_transform_tiling)
DEFINE_BENCHMARK(MANUL, _mlir_ciface_manul)
DEFINE_BENCHMARK(MANUL2, _mlir_ciface_manul2)
// DEFINE_BENCHMARK(MANUL3, _mlir_ciface_manul3)
} // namespace 

BENCHMARK(BM_GENERIC_ORIGIN)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GENERIC_TRANSFORM_TILING)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GENERIC_MANUL)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_GENERIC_MANUL2)->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_GENERIC_MANUL3)->Unit(benchmark::kMillisecond);

#define DEFINE_VERIFICATION(name, func) \
  void VERIFICATION_##name(MemRef<float, 2> input,  \
                           MemRef<float, 1> output, \
                           float resultScalar[]) { \
    func(&input, &output); \
    auto result##name = output.getData(); \
    std::cout << #name << "case: "  \
              << (areArraysEqual(resultScalar, result##name, 4096) \
                     ? PASS \
                     : FAIL) \
              << std::endl;\
  }

DEFINE_VERIFICATION(TRANSFORM_TILING, _mlir_ciface_transform_tiling)
DEFINE_VERIFICATION(MANUL, _mlir_ciface_manul)

void verification() {
  intptr_t sizesInput[2] = {INPUT_H, INPUT_W};
  intptr_t sizesOutput[1] = {OUTPUT_W};
  MemRef<float, 2> input(sizesInput, 1.0);
  MemRef<float, 1> output(sizesOutput, 1.0);
  _mlir_ciface_origin(&input, &output);
  auto resultScalar = output.getData();
  // Print the verification results.
  std::cout << "---------------------------------------------------------------"
               "---------"
            << std::endl;
  std::cout << "Correctness Verification:" << std::endl;
  MemRef<float, 1> newOutput1(sizesOutput, 1.0);
  VERIFICATION_TRANSFORM_TILING(input, newOutput1, resultScalar);
  MemRef<float, 1> newOutput2(sizesOutput, 1.0);
  VERIFICATION_MANUL(input, newOutput2, resultScalar);

  std::cout << "---------------------------------------------------------------"
               "---------"
            << std::endl;

}
