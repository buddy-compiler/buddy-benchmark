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
// This is the main file of the Halide Conv Layer benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <stdexcept>

void initializeHalideConvLayerBenchmark(char **);

void registerBenchmarkHalideConvLayer();

// Run benchmarks.
int main(int argc, char **argv) {
  if (argc != 1) {
    throw std::invalid_argument(
        "No arguments needed.\n");
  }

  initializeHalideConvLayerBenchmark(argv);

  // Register Benchmark Function.
  registerBenchmarkHalideConvLayer();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Generate result.

  return 0;
}
