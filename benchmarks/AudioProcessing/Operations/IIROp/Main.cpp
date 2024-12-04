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
// This is the main file of IIROp benchmark.
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
#define _MAX_ORDER 32
#define _PRINT true

// -----------------------------------------------------------------------------
// Global Variables and Functions. Please do not modify the code here.
// -----------------------------------------------------------------------------

univector<float, _IN_OUT_SIZE> iirInput;
univector<float, _IN_OUT_SIZE> iirOutput;
univector<float> kernel;
std::vector<biquad_params<float>> bqs;

intptr_t sizeofAud[1] = {_IN_OUT_SIZE};
intptr_t sizeofKernel[2] = {_MAX_ORDER, 6};

MemRef<float, 1> audRef(sizeofAud);
MemRef<float, 1> resRef(sizeofAud);
MemRef<float, 2> kernelRef(sizeofKernel);

using MLIRFunctionType = void (*)(MemRef<float, 1> *, MemRef<float, 2> *,
                                  MemRef<float, 1> *);

using BuddyFunctionType = void (*)(MemRef<float, 1> *, MemRef<float, 2> *,
                                   MemRef<float, 1> *, bool);

// Benchmarking function for MLIR based IIR method.
void DAP_OPS_IIR(benchmark::State &state, MLIRFunctionType func) {
  MemRef<float, 1> resRef(sizeofAud, 0.0);
  for (auto _ : state) {
    func(&audRef, &kernelRef, &resRef);
  }
  benchmark::DoNotOptimize(resRef);
}

void DAP_OPS_IIR(benchmark::State &state, BuddyFunctionType func,
                 bool isVectorization) {
  MemRef<float, 1> resRef(sizeofAud, 0.0);
  for (auto _ : state) {
    func(&audRef, &kernelRef, &resRef, isVectorization);
  }
  benchmark::DoNotOptimize(resRef);
}

// Benchmarking function for KFR IIR method.
static void KFR_IIR(benchmark::State &state) {
  for (auto _ : state) {
    iirOutput = biquad<_MAX_ORDER>(bqs, iirInput);
  }
}

// Verifies the result of an MLIR-based function against expected output.
void Verification(const univector<float, _IN_OUT_SIZE> &outputExpected,
                  MLIRFunctionType MLIRFunc, const std::string &name) {
  // Initialize MemRef with all zeros.
  MemRef<float, 1> outputGenerated(sizeofAud, 0.0);
  MLIRFunc(&audRef, &kernelRef, &outputGenerated);
  iirOp::printMemRef(outputGenerated, name, /*doPrint=*/_PRINT);
  iirOp::verify(outputExpected, outputGenerated, _IN_OUT_SIZE, name);
}

void Verification(const univector<float, _IN_OUT_SIZE> &outputExpected,
                  BuddyFunctionType BuddyFunc, bool isVectorization,
                  const std::string &name) {
  // Initialize MemRef with all zeros.
  MemRef<float, 1> outputGenerated(sizeofAud, 0.0);
  BuddyFunc(&audRef, &kernelRef, &outputGenerated, isVectorization);
  iirOp::printMemRef(outputGenerated, name, /*doPrint=*/_PRINT);
  iirOp::verify(outputExpected, outputGenerated, _IN_OUT_SIZE, name);
}

// -----------------------------------------------------------------------------
// Register Benchmark.
// -----------------------------------------------------------------------------

BENCHMARK_CAPTURE(DAP_OPS_IIR, mlir_scalar, _mlir_ciface_iir_scalar)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_IIR, buddy_scalar, dap::IIR<float, 1>, false)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_IIR, mlir_vectorize, _mlir_ciface_iir_vectorization)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_IIR, buddy_vectorize, dap::IIR<float, 1>, true)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK(KFR_IIR)->Unit(benchmark::kMillisecond)->Iterations(_NUM_ITER);

// -----------------------------------------------------------------------------
// Main Function.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize univectors and MemRefs.
  iirOp::initializeKFRIIR(iirInput, bqs, kernel);
  audRef = std::move(MemRef<float, 1>(iirInput.data(), sizeofAud));
  sizeofKernel[0] = bqs.size();
  kernelRef = std::move(MemRef<float, 2>(kernel.data(), sizeofKernel));
  resRef = std::move(MemRef<float, 1>(sizeofAud, 0.0));

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Obtain KFR output results as expected results in verification.
  iirOutput = biquad<_MAX_ORDER>(bqs, iirInput);
  iirOp::printUnivector(iirOutput, /*doPrint=*/_PRINT);

  // Verify the correctness of all methods.
  Verification(iirOutput, _mlir_ciface_iir_scalar, "MLIRScalar");
  Verification(iirOutput, dap::IIR<float, 1>, /*isVectorization=*/false,
               "BuddyScalar");
  Verification(iirOutput, _mlir_ciface_iir_vectorization, "MLIRVectorize");
  Verification(iirOutput, dap::IIR<float, 1>, /*isVectorization=*/true,
               "BuddyVectorize");

  return 0;
}
