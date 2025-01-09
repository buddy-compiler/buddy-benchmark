//===- Main.cpp -----------------------------------------------------------===//
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
// This is the main file of FIROp benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>
#include <type_traits>

using namespace std;

// -----------------------------------------------------------------------------
// Benchmark Configuration. Modify the numbers here for a custom benchmark.
// -----------------------------------------------------------------------------

#define _NUM_ITER 10
#define _IN_OUT_SIZE 2000000
#define _FILTER_SIZE 127
#define _PRINT false

// -----------------------------------------------------------------------------
// Global Variables and Functions. Please do not modify the code here.
// -----------------------------------------------------------------------------

univector<float, _IN_OUT_SIZE> firInput_f32, firOutput_f32;
univector<float, _FILTER_SIZE> firFilter_f32;
univector<double, _IN_OUT_SIZE> firInput_f64, firOutput_f64;
univector<double, _FILTER_SIZE> firFilter_f64;

intptr_t sizeofAud[1] = {_IN_OUT_SIZE};
intptr_t sizeofKernel[1] = {_FILTER_SIZE};

MemRef<float, 1> in_f32(sizeofAud), filt_f32(sizeofKernel), out_f32(sizeofAud);
MemRef<double, 1> in_f64(sizeofAud), filt_f64(sizeofKernel), out_f64(sizeofAud);

template <typename T>
using MLIRFunctionType = void (*)(MemRef<T, 1> *, MemRef<T, 1> *,
                                  MemRef<T, 1> *);

template <typename T>
using BuddyFunctionType = void (*)(MemRef<T, 1> *, MemRef<T, 1> *,
                                   MemRef<T, 1> *, bool);

// Benchmarking function for MLIR based FIR method.
template <typename T>
static void DAP_OPS_FIR(benchmark::State &state, MLIRFunctionType<T> func) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T must be either float (f32) or double (f64).");
  if constexpr (std::is_same<T, float>::value) {
    MemRef<T, 1> out_f32(sizeofAud, 0.0);
    for (auto _ : state) {
      func(&in_f32, &filt_f32, &out_f32);
    }
    benchmark::DoNotOptimize(out_f32);
  } else if constexpr (std::is_same<T, double>::value) {
    MemRef<T, 1> out_f64(sizeofAud, 0.0);
    for (auto _ : state) {
      func(&in_f64, &filt_f64, &out_f64);
    }
    benchmark::DoNotOptimize(out_f64);
  }
}

template <typename T>
static void DAP_OPS_FIR(benchmark::State &state, BuddyFunctionType<T> func,
                        bool isVectorization) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T must be either float (f32) or double (f64).");
  if constexpr (std::is_same<T, float>::value) {
    MemRef<T, 1> out_f32(sizeofAud, 0.0);
    for (auto _ : state) {
      func(&in_f32, &filt_f32, &out_f32, isVectorization);
    }
    benchmark::DoNotOptimize(out_f32);
  } else if constexpr (std::is_same<T, double>::value) {
    MemRef<T, 1> out_f64(sizeofAud, 0.0);
    for (auto _ : state) {
      func(&in_f64, &filt_f64, &out_f64, isVectorization);
    }
    benchmark::DoNotOptimize(out_f64);
  }
}

// Benchmarking function for KFR FIR method.
static void KFR_FIR_f32(benchmark::State &state) {
  for (auto _ : state) {
    firOutput_f32 = kfr::fir(firInput_f32, firFilter_f32);
  }
  benchmark::DoNotOptimize(firOutput_f32);
}

static void KFR_FIR_f64(benchmark::State &state) {
  for (auto _ : state) {
    firOutput_f64 = kfr::fir(firInput_f64, firFilter_f64);
  }
  benchmark::DoNotOptimize(firOutput_f64);
}

// Verifies the result of an MLIR-based function against expected output.
template <typename T>
void Verification(const univector<T, _IN_OUT_SIZE> &outputExpected,
                  MLIRFunctionType<T> MLIRFunc, const std::string &name) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T must be either float (f32) or double (f64).");
  // Initialize MemRef with all zeros.
  MemRef<T, 1> outputGenerated(sizeofAud, 0.0);
  if constexpr (std::is_same<T, float>::value) {
    MLIRFunc(&in_f32, &filt_f32, &outputGenerated);
  } else if constexpr (std::is_same<T, double>::value) {
    MLIRFunc(&in_f64, &filt_f64, &outputGenerated);
  }
  firOp::printMemRef(outputGenerated, name, /*doPrint=*/_PRINT);
  firOp::verify(outputExpected, outputGenerated, _IN_OUT_SIZE, name);
}

template <typename T>
void Verification(const univector<T, _IN_OUT_SIZE> &outputExpected,
                  BuddyFunctionType<T> BuddyFunc, bool isVectorization,
                  const std::string &name) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T must be either float (f32) or double (f64).");
  // Initialize MemRef with all zeros.
  MemRef<T, 1> outputGenerated(sizeofAud, 0.0);
  if constexpr (std::is_same<T, float>::value) {
    BuddyFunc(&in_f32, &filt_f32, &outputGenerated, isVectorization);
  } else if constexpr (std::is_same<T, double>::value) {
    BuddyFunc(&in_f64, &filt_f64, &outputGenerated, isVectorization);
  }
  firOp::printMemRef(outputGenerated, name, /*doPrint=*/_PRINT);
  firOp::verify(outputExpected, outputGenerated, _IN_OUT_SIZE, name);
}

// -----------------------------------------------------------------------------
// Register Benchmark.
// -----------------------------------------------------------------------------

// Benchmarks with f32/float type.
BENCHMARK_CAPTURE(DAP_OPS_FIR, buddy_scalar_f32, dap::FIR<float, 1>, false)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_FIR, mlir_vector_f32, _mlir_ciface_fir_vector_f32)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_FIR, buddy_tiled_vector_f32, dap::FIR<float, 1>, true)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK(KFR_FIR_f32)->Unit(benchmark::kMillisecond)->Iterations(_NUM_ITER);
// Benchmarks with f64/double type.
BENCHMARK_CAPTURE(DAP_OPS_FIR, buddy_scalar_f64, dap::FIR<double, 1>, false)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_FIR, mlir_vector_f64, _mlir_ciface_fir_vector_f64)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_FIR, buddy_tiled_vector_f64, dap::FIR<double, 1>,
                  true)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK(KFR_FIR_f64)->Unit(benchmark::kMillisecond)->Iterations(_NUM_ITER);

// -----------------------------------------------------------------------------
// Main Function.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize univectors and MemRefs.
  firOp::initializeKFRFIR(firInput_f32, firFilter_f32);
  in_f32 = std::move(MemRef<float, 1>(firInput_f32.data(), sizeofAud));
  filt_f32 = std::move(MemRef<float, 1>(firFilter_f32.data(), sizeofKernel));
  firOp::initializeKFRFIR(firInput_f64, firFilter_f64);
  in_f64 = std::move(MemRef<double, 1>(firInput_f64.data(), sizeofAud));
  filt_f64 = std::move(MemRef<double, 1>(firFilter_f64.data(), sizeofKernel));

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Obtain KFR output results as expected results in verification.
  firOutput_f32 = kfr::fir(firInput_f32, firFilter_f32);
  firOp::printUnivector(firOutput_f32, "KFRF32", /*doPrint=*/_PRINT);
  firOutput_f64 = kfr::fir(firInput_f64, firFilter_f64);
  firOp::printUnivector(firOutput_f64, "KFRF64", /*doPrint=*/_PRINT);

  // Verify the correctness of all methods.
  Verification(firOutput_f32, dap::FIR<float, 1>, /*isVectorization=*/false,
               "BuddyScalarF32");
  Verification(firOutput_f32, _mlir_ciface_fir_vector_f32, "MLIRVectorizeF32");
  Verification(firOutput_f32, dap::FIR<float, 1>, /*isVectorization=*/true,
               "BuddyTiledVectorizeF32");

  Verification(firOutput_f64, dap::FIR<double, 1>, /*isVectorization=*/false,
               "BuddyScalarF64");
  Verification(firOutput_f64, _mlir_ciface_fir_vector_f64, "MLIRVectorizeF64");
  Verification(firOutput_f64, dap::FIR<double, 1>, /*isVectorization=*/true,
               "BuddyTiledVectorizeF64");

  return 0;
}
