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
// This is the main file of the gccloops vectorization benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>

void generateResultMLIRGccLoopsEx1();
void generateResultMLIRGccLoopsEx2a();
void generateResultMLIRGccLoopsEx2b();
void generateResultMLIRGccLoopsEx3();
void generateResultMLIRGccLoopsEx4a();
void generateResultMLIRGccLoopsEx4b();
void generateResultMLIRGccLoopsEx4c();
void generateResultMLIRGccLoopsEx7();
void generateResultMLIRGccLoopsEx8();
void generateResultMLIRGccLoopsEx9();
void generateResultMLIRGccLoopsEx10a();
void generateResultMLIRGccLoopsEx10b();
void generateResultMLIRGccLoopsEx11();
void generateResultMLIRGccLoopsEx12();
void generateResultMLIRGccLoopsEx13();
void generateResultMLIRGccLoopsEx14();
void generateResultMLIRGccLoopsEx21();
void generateResultMLIRGccLoopsEx23();
void generateResultMLIRGccLoopsEx24();
void generateResultMLIRGccLoopsEx25();

// Run benchmarks.
int main(int argc, char **argv) {

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Generate result.
  generateResultMLIRGccLoopsEx1();
  generateResultMLIRGccLoopsEx2a();
  generateResultMLIRGccLoopsEx2b();
  generateResultMLIRGccLoopsEx3();
  generateResultMLIRGccLoopsEx4a();
  generateResultMLIRGccLoopsEx4b();
  generateResultMLIRGccLoopsEx4c();
  generateResultMLIRGccLoopsEx7();
  generateResultMLIRGccLoopsEx8();
  generateResultMLIRGccLoopsEx9();
  generateResultMLIRGccLoopsEx10a();
  generateResultMLIRGccLoopsEx10b();
  generateResultMLIRGccLoopsEx11();
  generateResultMLIRGccLoopsEx12();
  generateResultMLIRGccLoopsEx13();
  generateResultMLIRGccLoopsEx14();
  generateResultMLIRGccLoopsEx21();
  generateResultMLIRGccLoopsEx23();
  generateResultMLIRGccLoopsEx24();
  generateResultMLIRGccLoopsEx25();
  return 0;
}
