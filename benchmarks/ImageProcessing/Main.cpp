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
// This is the main file of the image processing benchmark.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <stdexcept>

void initializeMLIRConv2D(char **);
void initializeBuddyConv2D(char **);
void initializeBuddyCorr2D(char **);
void initializeOpenCVFilter2D(char **);
void initializeEigenConvolve2D(char **);

void generateResultBuddyConv2D(char **);
void generateResultBuddyCorr2D(char **);
void generateResultOpenCVFilter2D();
void generateResultEigenConvolve2D();

void registerBenchmarkBuddyCorr2D();
void registerBenchmarkOpenCVFilter2D();

// Run benchmarks.
int main(int argc, char **argv) {
  if (argc != 4) {
    throw std::invalid_argument(
        "Wrong format of command line arguments.\n"
        "Correct format is ./image-processing-benchmark <image path> <kernel "
        "name> <Boundary Option>\n where "
        "image path provides path of the image to be processed, kernel name "
        "denotes the name "
        "of desired kernel as specified in "
        "include/ImageProcessing/Kernels.h and Boundary options available "
        "are CONSTANT_PADDING, REPLICATE_PADDING.\n");
  }

  initializeMLIRConv2D(argv);
  initializeBuddyConv2D(argv);
  initializeBuddyCorr2D(argv);
  initializeOpenCVFilter2D(argv);
  initializeEigenConvolve2D(argv);

  registerBenchmarkBuddyCorr2D();
  registerBenchmarkOpenCVFilter2D();

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  // Generate result image.
  generateResultOpenCVFilter2D();
  generateResultBuddyConv2D(argv);
  generateResultBuddyCorr2D(argv);
  generateResultEigenConvolve2D();

  return 0;
}
