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
// This is the main file of BiquadOp benchmark.
//
//===----------------------------------------------------------------------===//

#include "Utils.hpp"
#include <benchmark/benchmark.h>

using namespace std;

// -----------------------------------------------------------------------------
// Benchmark Configuration. Modify the numbers here for a custom benchmark.
// -----------------------------------------------------------------------------

#define _NUM_ITER 10
#define _IN_OUT_SIZE 2000000
#define _PRINT true

// -----------------------------------------------------------------------------
// Global Variables and Functions. Please do not modify the code here.
// -----------------------------------------------------------------------------

biquad_params<float> bq;
univector<float, _IN_OUT_SIZE> biquadInput;
univector<float, _IN_OUT_SIZE> biquadOutput;
univector<float, 6> kernel;

intptr_t sizeofAud[1] = {_IN_OUT_SIZE};
intptr_t sizeofKernel[1] = {6};

MemRef<float, 1> audRef(sizeofAud);
MemRef<float, 1> resRef(sizeofAud);
MemRef<float, 1> kernelRef(sizeofKernel);

using MLIRFunctionType = void (*)(MemRef<float, 1> *, MemRef<float, 1> *,
                                  MemRef<float, 1> *);

// Benchmarking function for MLIR based Biquad method.
void DAP_OPS_Biquad(benchmark::State &state, MLIRFunctionType func) {
  MemRef<float, 1> resRef(sizeofAud, 0.0);
  for (auto _ : state) {
    func(&audRef, &kernelRef, &resRef);
  }
  benchmark::DoNotOptimize(resRef);
}

// Benchmarking function for KFR Biquad method.
static void KFR_Biquad(benchmark::State &state) {
  for (auto _ : state) {
    biquadOutput = kfr::biquad(bq, biquadInput);
  }
}

// Verifies the result of an MLIR-based function against expected output.
void Verification(const univector<float, _IN_OUT_SIZE> &outputExpected,
                  MLIRFunctionType MLIRFunc, const std::string &name) {
  // Initialize MemRef with all zeros.
  MemRef<float, 1> outputGenerated(sizeofAud, 0.0);
  MLIRFunc(&audRef, &kernelRef, &outputGenerated);
  biquadOp::printMemRef(outputGenerated, name, /*doPrint=*/_PRINT);
  biquadOp::verify(outputExpected, outputGenerated, _IN_OUT_SIZE, name);
}

// -----------------------------------------------------------------------------
// Register Benchmark.
// -----------------------------------------------------------------------------

BENCHMARK_CAPTURE(DAP_OPS_Biquad, mlir_biquad, _mlir_ciface_mlir_biquad)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_Biquad, buddy_biquad, dap::biquad<float, 1>)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK(KFR_Biquad)->Unit(benchmark::kMillisecond)->Iterations(_NUM_ITER);

// -----------------------------------------------------------------------------
// Main Function.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize univectors and MemRefs.
  biquadOp::initializeKFRBiquad(biquadInput, bq, kernel);
  audRef = std::move(MemRef<float, 1>(biquadInput.data(), sizeofAud));
  kernelRef = std::move(MemRef<float, 1>(kernel.data(), sizeofKernel));
  resRef = std::move(MemRef<float, 1>(sizeofAud, 0.0));

  // Run benchmarks.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Obtain KFR output results as expected results in verification.
  biquadOutput = kfr::biquad(bq, biquadInput);
  biquadOp::printUnivector(biquadOutput, /*doPrint=*/_PRINT);

  // Verify the correctness of all methods.
  Verification(biquadOutput, dap::biquad<float, 1>, "Buddy");
  Verification(biquadOutput, _mlir_ciface_mlir_biquad, "MLIR");

  return 0;
}
