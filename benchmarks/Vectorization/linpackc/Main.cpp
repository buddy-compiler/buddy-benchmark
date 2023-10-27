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
// This is the main file of the linpackc vectorization benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>

void generateResultMLIRLinpackCDaxpy();
void generateResultMLIRLinpackCMatgen();
void generateResultMLIRLinpackCDdot();
void generateResultMLIRLinpackCDscal();
void generateResultMLIRLinpackCIdamax();
void generateResultMLIRLinpackCDmxpy();
void generateResultMLIRLinpackCEpslon();

// Run benchmarks.
int main(int argc, char **argv) {

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Generate result.
  // generateResultMLIRLinpackCDaxpy();
  // generateResultMLIRLinpackCMatgen();
  // generateResultMLIRLinpackCDdot();
  // generateResultMLIRLinpackCDscal();
  // generateResultMLIRLinpackCIdamax();
  // generateResultMLIRLinpackCDmxpy();
  // generateResultMLIRLinpackCEpslon();
  return 0;
}
