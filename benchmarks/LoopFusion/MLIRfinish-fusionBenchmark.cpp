//===- MLIRlogBenchmark.cpp --------------------------------------------===//
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

#include <iostream>
#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>

extern "C"{
    void _mlir_ciface_main2();
}

static void MLIR_fuseloop(benchmark::State &state){
    for (auto _ : state) {
        for (int i = 0; i < state.range(0); ++i) {
            _mlir_ciface_main2();
        }
    }
}

// Register benchmarking function.
BENCHMARK(MLIR_fuseloop)->Arg(1);

// Generate result
void generateResultMLIR_fuse(){
    _mlir_ciface_main2();
    std::cout << "--------------------------------------------------------"
            << std::endl;
    std::cout << "MLIR_fuseloop has no output for now" << std::endl;
}