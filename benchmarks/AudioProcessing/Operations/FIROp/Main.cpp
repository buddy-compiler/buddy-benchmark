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

using namespace std;

// -----------------------------------------------------------------------------
// Benchmark Configuration. Modify the numbers here for a custom benchmark.
// -----------------------------------------------------------------------------

#define _NUM_ITER 10
#define _IN_OUT_SIZE 2000000
#define _KERNEL_SIZE 127
#define _PRINT true

// -----------------------------------------------------------------------------
// Global Variables and Functions. Please do not modify the code here.
// -----------------------------------------------------------------------------

univector<float, _IN_OUT_SIZE> firInput;
univector<float, _IN_OUT_SIZE> firOutput;
univector<float, _KERNEL_SIZE> taps127;

intptr_t sizeofAud[1] = {_IN_OUT_SIZE};
intptr_t sizeofKernel[1] = {_KERNEL_SIZE};

MemRef<float, 1> audRef(sizeofAud);
MemRef<float, 1> resRef(sizeofAud);
MemRef<float, 1> kernelRef(sizeofKernel);

using MLIRFunctionType = void (*)(MemRef<float, 1> *, MemRef<float, 1> *,
                                  MemRef<float, 1> *);

// Benchmarking function for MLIR based FIR method.
void DAP_OPS_FIR(benchmark::State &state, MLIRFunctionType func) {
  MemRef<float, 1> resRef(sizeofAud, 0.0);
  for (auto _ : state) {
    func(&audRef, &kernelRef, &resRef);
  }
  benchmark::DoNotOptimize(resRef);
}

// Benchmarking function for KFR FIR method.
static void KFR_FIR(benchmark::State &state) {
  for (auto _ : state) {
    firOutput = kfr::fir(firInput, taps127);
  }
}

// Verifies the result of an MLIR-based function against expected output.
void Verification(const univector<float, _IN_OUT_SIZE> &outputExpected,
                  MLIRFunctionType MLIRFunc, const std::string &name) {
  // Initialize MemRef with all zeros.
  MemRef<float, 1> outputGenerated(sizeofAud, 0.0);
  MLIRFunc(&audRef, &kernelRef, &outputGenerated);
  firOp::printMemRef(outputGenerated, name, /*doPrint=*/_PRINT);
  firOp::verify(outputExpected, outputGenerated, _IN_OUT_SIZE, name);
}

// -----------------------------------------------------------------------------
// Register Benchmark.
// -----------------------------------------------------------------------------

BENCHMARK_CAPTURE(DAP_OPS_FIR, linalg_conv1d, _mlir_ciface_mlir_linalg_conv1d)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK_CAPTURE(DAP_OPS_FIR, buddy_fir, dap::FIR<float, 1>)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(_NUM_ITER);
BENCHMARK(KFR_FIR)->Unit(benchmark::kMillisecond)->Iterations(_NUM_ITER);

// -----------------------------------------------------------------------------
// Main Function.
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
  // Initialize univectors and MemRefs.
  firOp::initializeKFRFIR(firInput, taps127);
  audRef = std::move(MemRef<float, 1>(firInput.data(), sizeofAud));
  kernelRef = std::move(MemRef<float, 1>(taps127.data(), sizeofKernel));
  resRef = std::move(MemRef<float, 1>(sizeofAud, 0.0));

  // Run benchmark.
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  std::cout << "\033[34m---------- Verification ----------\033[0m" << std::endl;
  // Obtain KFR output results as expected results in verification.
  firOutput = kfr::fir(firInput, taps127);
  firOp::printUnivector(firOutput, /*doPrint=*/_PRINT);

  // Verify the correctness of all methods.
  Verification(firOutput, dap::FIR<float, 1>, "Buddy");
  Verification(firOutput, _mlir_ciface_mlir_linalg_conv1d, "MLIR");

  return 0;
}
